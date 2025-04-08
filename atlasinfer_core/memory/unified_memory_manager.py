# atlasinfer_core/memory/unified_memory_manager.py
import torch
import lmdb
import numpy as np
import os
import io # For serialization
from collections import OrderedDict
from typing import Tuple, Optional, Dict, Any
import threading
import queue
import time
import gc # Garbage collection

# --- LMDB Interface (Helper Class) ---
class LMDBInterface:
    """ Simple wrapper for LMDB database interactions. """
    def __init__(self, path: str, map_size_gb: int = 50):
        self.path = path
        self.env = None # Initialize later
        self.map_size = int(map_size_gb * (1024**3)) # GB to bytes

        try:
            # Ensure directory exists
            os.makedirs(path, exist_ok=True)
            # Open LMDB environment
            self.env = lmdb.open(path, map_size=self.map_size, subdir=True,
                                 readonly=False, lock=False, readahead=False, meminit=False)
            print(f"LMDB disk cache initialized at: {path}")
        except lmdb.Error as e:
            print(f"CRITICAL: Error opening LMDB environment at {path}: {e}")
            print("LMDB cache might be corrupted or path invalid. Try deleting the directory.")
            # Decide how to handle: raise error, or proceed without disk cache?
            self.env = None # Indicate failure
            print("Warning: Proceeding without Disk Cache due to LMDB error.")
        except Exception as e:
            print(f"CRITICAL: Unexpected error initializing LMDB: {e}")
            self.env = None
            print("Warning: Proceeding without Disk Cache.")


    def _serialize_tensor(self, tensor: torch.Tensor) -> bytes:
        """ Serializes tensor to bytes using torch.save """
        buffer = io.BytesIO()
        torch.save(tensor.cpu(), buffer) # Always save from CPU
        return buffer.getvalue()

    def _deserialize_tensor(self, buffer: bytes) -> Optional[torch.Tensor]:
        """ Deserializes tensor from bytes using torch.load """
        buffer = io.BytesIO(buffer)
        try:
            # Load directly to CPU
            tensor = torch.load(buffer, map_location='cpu')
            return tensor
        except Exception as e:
            print(f"Error deserializing tensor from LMDB: {e}")
            return None # Indicate failure

    def put(self, key: str, tensor: torch.Tensor) -> bool:
        """ Stores a tensor in LMDB. Returns True on success. """
        if self.env is None: return False # Cannot store if env failed
        key_bytes = key.encode('utf-8')
        value_bytes = self._serialize_tensor(tensor)
        try:
            with self.env.begin(write=True) as txn:
                success = txn.put(key_bytes, value_bytes)
            return success
        except lmdb.MapFullError:
            print(f"LMDB MapFullError: Cannot write key '{key}'. Increase disk_limit_gb or clear cache.")
            # TODO: Could implement automatic resizing or LRU eviction within LMDB itself (more complex)
            return False
        except lmdb.Error as e:
            print(f"LMDB Error putting key '{key}': {e}")
            return False

    def get(self, key: str) -> Optional[torch.Tensor]:
        """ Retrieves a tensor from LMDB. """
        if self.env is None: return None
        key_bytes = key.encode('utf-8')
        try:
            with self.env.begin(write=False) as txn:
                value_bytes = txn.get(key_bytes)
                if value_bytes:
                    return self._deserialize_tensor(value_bytes)
                else:
                    return None
        except lmdb.Error as e:
            print(f"LMDB Error getting key '{key}': {e}")
            return None

    def delete(self, key: str) -> bool:
        """ Deletes a key from LMDB. Returns True on success. """
        if self.env is None: return False
        key_bytes = key.encode('utf-8')
        try:
            with self.env.begin(write=True) as txn:
                success = txn.delete(key_bytes)
            return success
        except lmdb.Error as e:
            print(f"LMDB Error deleting key '{key}': {e}")
            return False

    def close(self):
        """ Closes the LMDB environment. """
        if self.env:
            self.env.close()
            print("LMDB disk cache closed.")
            self.env = None


# --- Unified Memory Manager ---
class UnifiedMemoryManager:
    def __init__(self,
                 vram_limit_gb: float, # Target VRAM limit for KV cache
                 ram_limit_gb: float,  # Target RAM limit for KV cache
                 disk_path: Optional[str], # Path for LMDB, None to disable disk
                 disk_limit_gb: int = 50,
                 gpu_device: torch.device = torch.device("cuda:0"),
                 cpu_device: torch.device = torch.device("cpu"),
                 vram_only_kv_cache: bool = False): # Try to keep KV only in VRAM

        self.gpu_device = gpu_device if torch.cuda.is_available() else cpu_device
        self.cpu_device = cpu_device
        self.can_use_gpu = (self.gpu_device != self.cpu_device)

        # Calculate capacities in bytes
        self.vram_capacity = int(vram_limit_gb * (1024**3)) if self.can_use_gpu else 0
        self.ram_capacity = int(ram_limit_gb * (1024**3))

        self.vram_only_mode = vram_only_kv_cache if self.can_use_gpu else True # Force vram_only if no GPU

        # Caches using OrderedDict for LRU behavior
        # Key: String (e.g., "kv_layer_0_K"), Value: Tensor on respective device
        self.vram_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.ram_cache: OrderedDict[str, torch.Tensor] = OrderedDict()

        # Disk Interface (conditional)
        self.disk_interface = None
        self.use_disk = False
        if disk_path and disk_path.lower() != 'none' and not self.vram_only_mode:
            self.disk_interface = LMDBInterface(disk_path, map_size_gb=disk_limit_gb)
            if self.disk_interface.env is not None: # Check if LMDB init succeeded
                 self.use_disk = True
            else:
                 print("Disabling disk cache due to LMDB initialization failure.")
                 self.disk_interface = None # Ensure it's None if failed

        # Current usage trackers
        self.vram_used = 0
        self.ram_used = 0

        # Statistics
        self.stats = {
            "vram_hits": 0, "ram_hits": 0, "disk_hits": 0, "misses": 0,
            "evict_vram": 0, "evict_ram": 0, "load_ram": 0, "load_disk": 0,
            "put_vram": 0, "put_ram": 0, "put_disk": 0
        }
        self.stats_lock = threading.Lock() # Protect stats updates

        # Background writer setup (optional, start simple)
        self.async_write = False # Control flag
        self.disk_write_queue = queue.Queue()
        self.disk_writer_thread = None
        if self.use_disk and self.async_write:
             self.disk_writer_thread = threading.Thread(target=self._disk_writer_loop, daemon=True)
             self.disk_writer_thread.start()

        print("Unified Memory Manager initialized.")
        print(f"  VRAM Capacity: {self.vram_capacity / (1024**3):.2f} GB")
        print(f"  RAM Capacity: {self.ram_capacity / (1024**3):.2f} GB")
        print(f"  Disk Cache Enabled: {self.use_disk}")
        print(f"  VRAM-Only Mode (KV Cache): {self.vram_only_mode}")


    def _get_tensor_size(self, tensor: torch.Tensor) -> int:
        """ Returns tensor size in bytes. """
        return tensor.element_size() * tensor.nelement()

    def _update_stat(self, key, value=1):
        """ Safely updates statistics. """
        with self.stats_lock:
            self.stats[key] += value

    def _evict_lru_vram_to_ram(self) -> bool:
        """ Evicts LRU item from VRAM to RAM. Returns True if eviction happened. """
        if not self.vram_cache: return False
        key, tensor_gpu = self.vram_cache.popitem(last=False) # Pop LRU
        size = self._get_tensor_size(tensor_gpu)
        self.vram_used -= size
        self._update_stat("evict_vram")

        if self.vram_only_mode:
            # print(f"DEBUG: VRAM-Only Mode: Deleting evicted VRAM tensor {key}")
            del tensor_gpu
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            return True

        # --- Move to RAM ---
        tensor_cpu = tensor_gpu.to(self.cpu_device)
        del tensor_gpu # Explicit delete helps GPU memory free faster
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # Make space in RAM if needed
        while self.ram_used + size > self.ram_capacity and self.ram_cache:
            if not self._evict_lru_ram_to_disk():
                 print(f"ERROR: Failed to evict from RAM to make space for {key}. Data might be lost.")
                 del tensor_cpu # Delete the tensor that couldn't fit
                 return False # Failed to place in RAM

        # Add to RAM cache
        self.ram_cache[key] = tensor_cpu
        self.ram_used += size
        self.ram_cache.move_to_end(key) # Mark as recently used in RAM
        # print(f"DEBUG: Evicted {key} from VRAM to RAM.")
        return True


    def _evict_lru_ram_to_disk(self) -> bool:
        """ Evicts LRU item from RAM to Disk. Returns True if eviction happened. """
        if not self.ram_cache or not self.use_disk: return False
        key, tensor_cpu = self.ram_cache.popitem(last=False) # Pop LRU
        size = self._get_tensor_size(tensor_cpu)
        self.ram_used -= size
        self._update_stat("evict_ram")

        # --- Write to Disk ---
        if self.async_write:
            self.disk_write_queue.put((key, tensor_cpu))
            # print(f"DEBUG: Queued {key} for async disk write.")
        else: # Synchronous write
            success = self.disk_interface.put(key, tensor_cpu)
            if not success:
                 print(f"Warning: Failed to write evicted RAM item {key} to disk.")
                 # Item is lost if write fails synchronously
            # print(f"DEBUG: Evicted {key} from RAM to Disk (Sync).")
            del tensor_cpu # Free CPU memory now
            gc.collect()

        return True

    def _load_ram_to_vram(self, key: str) -> Optional[torch.Tensor]:
        """ Loads item from RAM to VRAM, returns tensor on GPU or None if failed. """
        if not self.can_use_gpu: return None # Cannot load to VRAM if no GPU
        if key not in self.ram_cache: return None

        tensor_cpu = self.ram_cache.pop(key)
        size = self._get_tensor_size(tensor_cpu)
        self.ram_used -= size
        self._update_stat("load_ram")

        # Make space in VRAM if needed
        made_space = True
        while self.vram_used + size > self.vram_capacity and self.vram_cache:
             if not self._evict_lru_vram_to_ram():
                  print(f"ERROR: Failed to evict from VRAM to make space for {key}.")
                  made_space = False
                  break # Stop trying if eviction fails

        if not made_space:
             # Put item back in RAM if we failed to make space
             self.ram_cache[key] = tensor_cpu
             self.ram_used += size
             print(f"ERROR: Could not load {key} to VRAM (no space). Kept in RAM.")
             return None

        # Move to VRAM
        try:
            tensor_gpu = tensor_cpu.to(self.gpu_device)
            del tensor_cpu # Free CPU RAM
            gc.collect()

            # Add to VRAM cache
            self.vram_cache[key] = tensor_gpu
            self.vram_used += size
            self.vram_cache.move_to_end(key) # Mark as recently used
            # print(f"DEBUG: Loaded {key} from RAM to VRAM.")
            return tensor_gpu
        except Exception as e: # Catch potential CUDA OOM during move
             print(f"ERROR: Failed to move tensor {key} to VRAM: {e}")
             # Try putting it back in RAM?
             self.ram_cache[key] = tensor_cpu
             self.ram_used += size
             return None


    def _load_disk_to_ram(self, key: str) -> Optional[torch.Tensor]:
        """ Loads item from Disk to RAM, returns tensor on CPU or None if failed. """
        if not self.use_disk: return None

        tensor_cpu = self.disk_interface.get(key)
        if tensor_cpu is None:
            # print(f"DEBUG: Key {key} not found on disk.")
            return None # Not found on disk
        self._update_stat("load_disk")

        size = self._get_tensor_size(tensor_cpu)

        # Make space in RAM if needed
        made_space = True
        while self.ram_used + size > self.ram_capacity and self.ram_cache:
             if not self._evict_lru_ram_to_disk():
                  print(f"ERROR: Failed to evict from RAM to make space for disk load {key}.")
                  made_space = False
                  break

        if not made_space:
             print(f"ERROR: Could not load {key} to RAM (no space). Item remains on disk.")
             del tensor_cpu # Delete the loaded tensor
             gc.collect()
             return None

        # Add to RAM cache
        self.ram_cache[key] = tensor_cpu
        self.ram_used += size
        self.ram_cache.move_to_end(key) # Mark as recently used
        # Optionally delete from disk after successful load to RAM?
        # self.disk_interface.delete(key)
        # print(f"DEBUG: Loaded {key} from Disk to RAM.")
        return tensor_cpu

    # --- KV Cache Specific Public Methods ---

    def _get_kv_key(self, layer_idx: int, type: str) -> str:
        """ Generates standardized key for K or V tensor. """
        return f"kv_{layer_idx}_{type.upper()}" # e.g., kv_0_K, kv_1_V

    def put_kv(self, layer_idx: int, k_tensor: torch.Tensor, v_tensor: torch.Tensor):
        """
        Stores K and V tensors for a layer (full history). Assumes input on GPU.
        Overwrites existing entries for the layer. Tries VRAM -> RAM -> Disk.
        """
        if not self.can_use_gpu and k_tensor.device != self.cpu_device:
             # If no GPU, input must be CPU
             k_tensor = k_tensor.to(self.cpu_device)
             v_tensor = v_tensor.to(self.cpu_device)
        elif self.can_use_gpu and k_tensor.device != self.gpu_device:
             # If GPU exists, input should be on GPU
             k_tensor = k_tensor.to(self.gpu_device)
             v_tensor = v_tensor.to(self.gpu_device)

        keys = [self._get_kv_key(layer_idx, 'K'), self._get_kv_key(layer_idx, 'V')]
        tensors = [k_tensor, v_tensor]

        for key, tensor in zip(keys, tensors):
            size = self._get_tensor_size(tensor)

            # --- Remove existing entry if present (to update size correctly) ---
            if key in self.vram_cache:
                old = self.vram_cache.pop(key)
                self.vram_used -= self._get_tensor_size(old)
                del old
            if key in self.ram_cache:
                 old = self.ram_cache.pop(key)
                 self.ram_used -= self._get_tensor_size(old)
                 del old
            if self.use_disk:
                 # No need to delete from disk immediately, overwrite is fine
                 pass

            # --- Try placing in VRAM (if GPU available) ---
            placed = False
            if self.can_use_gpu:
                # Make space
                while self.vram_used + size > self.vram_capacity and self.vram_cache:
                    if not self._evict_lru_vram_to_ram(): break # Stop if eviction fails
                # Try placing
                if self.vram_used + size <= self.vram_capacity:
                    self.vram_cache[key] = tensor # Assume tensor is already on GPU
                    self.vram_used += size
                    self.vram_cache.move_to_end(key)
                    self._update_stat("put_vram")
                    placed = True
                    # print(f"DEBUG: Placed {key} in VRAM.")

            # --- Try placing in RAM (if not placed in VRAM or VRAM-only mode disabled) ---
            if not placed and not self.vram_only_mode:
                 # Ensure tensor is on CPU for RAM cache
                 tensor_cpu = tensor.to(self.cpu_device) if tensor.device != self.cpu_device else tensor

                 # Make space
                 while self.ram_used + size > self.ram_capacity and self.ram_cache:
                      if not self._evict_lru_ram_to_disk(): break
                 # Try placing
                 if self.ram_used + size <= self.ram_capacity:
                      self.ram_cache[key] = tensor_cpu
                      self.ram_used += size
                      self.ram_cache.move_to_end(key)
                      self._update_stat("put_ram")
                      placed = True
                      # print(f"DEBUG: Placed {key} in RAM.")
                 # If tensor_cpu was created, and original tensor was on GPU, delete cpu copy if not placed
                 if tensor_cpu is not tensor and not placed:
                      del tensor_cpu


            # --- Try placing on Disk (if not placed elsewhere and disk enabled) ---
            if not placed and self.use_disk:
                 tensor_cpu = tensor.to(self.cpu_device) if tensor.device != self.cpu_device else tensor
                 # No capacity check needed for disk put (LMDB handles map size)
                 if self.async_write:
                      self.disk_write_queue.put((key, tensor_cpu))
                 else:
                      if self.disk_interface.put(key, tensor_cpu):
                          self._update_stat("put_disk")
                          placed = True
                          # print(f"DEBUG: Placed {key} on Disk.")
                      else:
                          print(f"Warning: Failed to place {key} on disk.")

                 # If tensor_cpu was created, delete if not same as input
                 if tensor_cpu is not tensor:
                      del tensor_cpu

            # --- Final check ---
            # if not placed: print(f"Warning: Failed to place KV tensor {key} in any cache tier.")


    def get_kv(self, layer_idx: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Retrieves K and V history for a layer, loading up the hierarchy.
        Returns tensors on the GPU device (if available), otherwise CPU.
        Returns (None, None) if not found or error occurs.
        """
        key_k = self._get_kv_key(layer_idx, 'K')
        key_v = self._get_kv_key(layer_idx, 'V')
        k_tensor_target, v_tensor_target = None, None
        target_device = self.gpu_device # Target device is GPU if possible

        for i, key in enumerate([key_k, key_v]):
            tensor_target = None
            found_in = None # Track where it was found ('vram', 'ram', 'disk')

            # 1. Check VRAM
            if self.can_use_gpu and key in self.vram_cache:
                self._update_stat("vram_hits")
                self.vram_cache.move_to_end(key)
                tensor_target = self.vram_cache[key] # Already on target_device (GPU)
                found_in = 'vram'
                # print(f"DEBUG: Cache hit VRAM for {key}")

            # 2. Check RAM
            elif not self.vram_only_mode and key in self.ram_cache:
                 self._update_stat("ram_hits")
                 self.ram_cache.move_to_end(key) # Mark used in RAM first
                 # Try loading RAM -> VRAM
                 loaded_to_vram = self._load_ram_to_vram(key)
                 if loaded_to_vram is not None:
                      tensor_target = loaded_to_vram
                      found_in = 'ram' # Found in RAM, now promoted to VRAM
                      # print(f"DEBUG: Cache hit RAM for {key}, loaded to VRAM")
                 elif self.can_use_gpu: # Failed to load to VRAM
                      print(f"Warning: Failed loading {key} from RAM->VRAM. Trying CPU access.")
                      # Try to return the CPU tensor directly if no GPU needed or available
                      tensor_target = self.ram_cache.get(key) # Get without popping
                      if tensor_target is not None:
                           target_device = self.cpu_device # Fallback to CPU processing maybe?
                           found_in = 'ram_cpu'
                 else: # No GPU, just return from RAM (already on CPU)
                      tensor_target = self.ram_cache.get(key)
                      found_in = 'ram'


            # 3. Check Disk
            elif self.use_disk: # Only check disk if enabled
                 # Load Disk -> RAM first
                 loaded_to_ram = self._load_disk_to_ram(key)
                 if loaded_to_ram is not None:
                      self._update_stat("disk_hits")
                      # Now try loading RAM -> VRAM
                      loaded_to_vram = self._load_ram_to_vram(key) # key is now in ram_cache
                      if loaded_to_vram is not None:
                           tensor_target = loaded_to_vram
                           found_in = 'disk' # Found on disk, now promoted to VRAM
                           # print(f"DEBUG: Cache hit Disk for {key}, loaded to VRAM")
                      elif self.can_use_gpu:
                           print(f"Warning: Failed loading {key} from RAM->VRAM after disk load.")
                           tensor_target = self.ram_cache.get(key) # Try CPU access
                           if tensor_target is not None:
                                target_device = self.cpu_device
                                found_in = 'disk_cpu'
                      else: # No GPU
                           tensor_target = self.ram_cache.get(key) # Return from RAM
                           found_in = 'disk'


            # 4. Miss
            if found_in is None:
                 # print(f"DEBUG: Cache miss for {key}")
                 self._update_stat("misses")
                 # Leave tensor_target as None

            # Assign to k_tensor or v_tensor
            if i == 0: k_tensor_target = tensor_target
            else: v_tensor_target = tensor_target

        # Final consistency check: both should be None or both Tensors
        if (k_tensor_target is None) != (v_tensor_target is None):
             print(f"CRITICAL: Inconsistent KV cache state for layer {layer_idx}. K={k_tensor_target is not None}, V={v_tensor_target is not None}. Returning None.")
             # Attempt cleanup? Maybe remove the one that exists?
             return None, None

        # Ensure both are on the final target device (GPU or CPU fallback)
        if k_tensor_target is not None and k_tensor_target.device != target_device:
             k_tensor_target = k_tensor_target.to(target_device)
        if v_tensor_target is not None and v_tensor_target.device != target_device:
             v_tensor_target = v_tensor_target.to(target_device)

        return k_tensor_target, v_tensor_target


    def get_stats(self) -> Dict[str, Any]:
        """ Returns a dictionary of cache statistics. """
        with self.stats_lock:
             # Make a copy to avoid returning internal dict
             current_stats = self.stats.copy()

        current_stats["vram_used_gb"] = self.vram_used / (1024**3)
        current_stats["ram_used_gb"] = self.ram_used / (1024**3)
        current_stats["vram_cache_items"] = len(self.vram_cache)
        current_stats["ram_cache_items"] = len(self.ram_cache)
        current_stats["total_hits"] = current_stats["vram_hits"] + current_stats["ram_hits"] + current_stats["disk_hits"]

        # Estimate disk usage
        current_stats["disk_used_gb_est"] = "N/A (Disk Disabled)"
        # Check if disk is enabled AND interface object exists AND lmdb env is open
        if self.use_disk and self.disk_interface and self.disk_interface.env:
             try:
                  # Use env.stat() which should be available
                  db_stat = self.disk_interface.env.stat()
                  # last_pgno is the last page number used; psize is page size
                  disk_bytes = db_stat['psize'] * (db_stat['last_pgno'] + 1) # Estimate based on last page used
                  current_stats["disk_used_gb_est"] = f"{disk_bytes / (1024**3):.3f}"
             except Exception as e:
                  # Optional: print(f"Debug: Error reading LMDB stats - {e}")
                  current_stats["disk_used_gb_est"] = "Error reading stats" # Keep error message concise

        return current_stats

    def close(self):
        """ Closes disk interface and potentially stops threads. """
        if self.disk_interface:
            self.disk_interface.close()
        # Add logic to signal writer thread to stop if implemented
        # self.disk_write_queue.put(None) # Sentinel value
        # if self.disk_writer_thread: self.disk_writer_thread.join()

    # --- Optional: Background Disk Writer ---
    def _disk_writer_loop(self):
        """ Background thread to write evicted RAM items to disk. """
        while True:
            item = self.disk_write_queue.get() # Blocks until item available
            if item is None: # Sentinel value to stop
                break
            key, tensor_cpu = item
            success = self.disk_interface.put(key, tensor_cpu)
            if success:
                 self._update_stat("put_disk")
                 # print(f"DEBUG: Async write completed for {key}")
            else:
                 print(f"Warning: Async disk write FAILED for key {key}.")
            del tensor_cpu # Free memory after attempting write
            gc.collect()
            self.disk_write_queue.task_done() # Signal completion