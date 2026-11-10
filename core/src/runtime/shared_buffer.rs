//! Reference-counted allocation for `SharedArrayBuffer` data.
//!
//! A `SharedArrayBuffer`'s data block is shared, not copied
//! (<https://tc39.es/ecma262/#sec-abstract-operations-for-sharedarraybuffer-objects>).
//! Two QuickJS runtimes on different threads can view the same block only if
//! they allocate it with the same allocator, because QuickJS's default
//! allocator belongs to one runtime. Installing these hooks at runtime
//! construction makes every `SharedArrayBuffer` block reference-counted, so a
//! second runtime can wrap an existing pointer with
//! `JS_NewArrayBuffer(..., is_shared = true)` and keep it alive through
//! `sab_dup`.
//!
//! The hooks also make growable `SharedArrayBuffer` possible, which QuickJS
//! refuses without them.

use alloc::alloc::{alloc, dealloc};
use core::alloc::Layout;
use core::ffi::c_void;
use core::mem;
use core::ptr;
use core::sync::atomic::{AtomicUsize, Ordering};

use crate::qjs;
use crate::runtime::Runtime;

/// Bytes in front of every block: the reference count and the block size.
const HEADER: usize = mem::size_of::<usize>() * 2;
/// Alignment of array-buffer data. The header keeps the returned pointer
/// aligned.
const ALIGN: usize = 16;

fn layout(size: usize) -> Option<Layout> {
    Layout::from_size_align(HEADER.checked_add(size)?, ALIGN).ok()
}

/// The header start for a user pointer.
///
/// # Safety
/// `ptr` came from [`sab_alloc`], so `HEADER` bytes precede it.
unsafe fn base(ptr: *mut c_void) -> *mut u8 {
    (ptr as *mut u8).sub(HEADER)
}

/// # Safety
/// Same as [`base`].
unsafe fn refs(ptr: *mut c_void) -> &'static AtomicUsize {
    &*(base(ptr) as *const AtomicUsize)
}

/// # Safety
/// Same as [`base`].
unsafe fn block_size(ptr: *mut c_void) -> usize {
    *((base(ptr).add(mem::size_of::<usize>())) as *const usize)
}

/// The number of bytes this allocator committed for `ptr`'s block.
///
/// A growable `SharedArrayBuffer` commits its maximum size upfront, while
/// `JS_GetArrayBuffer` reports the current byte length, so the header is the
/// only place the allocation size survives.
///
/// # Safety
/// `ptr` must be a `SharedArrayBuffer` backing store allocated after
/// [`Runtime::install_shared_array_buffer_hooks`], so the header exists.
pub(crate) unsafe fn allocation_size(ptr: *mut c_void) -> usize {
    // SAFETY: the caller guarantees the pointer came from `sab_alloc`.
    unsafe { block_size(ptr) }
}

unsafe extern "C" fn sab_alloc(_opaque: *mut c_void, size: qjs::size_t) -> *mut c_void {
    let size = size as usize;
    let Some(layout) = layout(size) else {
        return ptr::null_mut();
    };
    // SAFETY: the layout has a non-zero size (`HEADER` alone), so `alloc` is
    // valid; a null result is the failure signal QuickJS expects.
    let base = unsafe { alloc(layout) };
    if base.is_null() {
        return ptr::null_mut();
    }
    // SAFETY: the block is at least `HEADER` bytes and 16-aligned, so both
    // header fields fit.
    unsafe {
        (base as *mut AtomicUsize).write(AtomicUsize::new(1));
        (base.add(mem::size_of::<usize>()) as *mut usize).write(size);
        base.add(HEADER) as *mut c_void
    }
}

unsafe extern "C" fn sab_free(_opaque: *mut c_void, ptr: *mut c_void) {
    if ptr.is_null() {
        return;
    }
    // SAFETY: the pointer came from `sab_alloc`, so the header is intact.
    let refs = unsafe { refs(ptr) };
    if refs.fetch_sub(1, Ordering::AcqRel) != 1 {
        return;
    }
    let size = unsafe { block_size(ptr) };
    if let Some(layout) = layout(size) {
        // SAFETY: the last reference is gone, and `layout` is the one the
        // block was allocated with.
        unsafe { dealloc(base(ptr), layout) };
    }
}

unsafe extern "C" fn sab_dup(_opaque: *mut c_void, ptr: *mut c_void) {
    if ptr.is_null() {
        return;
    }
    // SAFETY: the pointer came from `sab_alloc`, so the header is intact.
    unsafe { refs(ptr).fetch_add(1, Ordering::Relaxed) };
}

const FUNCTIONS: qjs::JSSharedArrayBufferFunctions = qjs::JSSharedArrayBufferFunctions {
    sab_alloc: Some(sab_alloc),
    sab_free: Some(sab_free),
    sab_dup: Some(sab_dup),
    sab_opaque: ptr::null_mut::<c_void>(),
};

/// Takes one reference to a block this allocator produced.
pub(crate) fn retain(ptr: *mut u8) {
    // SAFETY: the caller obtained `ptr` from a `SharedArrayBuffer` allocated
    // after the hooks were installed, so the header is intact.
    unsafe { sab_dup(ptr::null_mut(), ptr as *mut c_void) };
}

/// Releases one reference to a block this allocator produced.
pub(crate) fn release(ptr: *mut u8) {
    // SAFETY: same as [`retain`].
    unsafe { sab_free(ptr::null_mut(), ptr as *mut c_void) };
}

/// The QuickJS class id of a `SharedArrayBuffer`, discovered once per process.
///
/// QuickJS's class ids are internal, so the value is read from a real object
/// instead of copied from an enum: an upstream renumbering then cannot make a
/// plain `ArrayBuffer` look shared. The probe allocates through the hooks so
/// its finalizer's `sab_free` matches its `sab_alloc`.
pub(crate) fn shared_class_id(ctx: *mut qjs::JSContext) -> u32 {
    /// `0` is `JS_INVALID_CLASS_ID`, so it doubles as "not measured yet".
    static CLASS: AtomicUsize = AtomicUsize::new(0);
    let known = CLASS.load(Ordering::Acquire);
    if known != 0 {
        return known as u32;
    }
    // SAFETY: the allocator is installed and the context is live.
    let ptr = unsafe { sab_alloc(ptr::null_mut(), 1) };
    if ptr.is_null() {
        return u32::MAX;
    }
    // SAFETY: `ptr` is a hook allocation of one byte; the call takes its
    // own reference and the value owns that reference until it is freed.
    let value = unsafe {
        qjs::JS_NewArrayBuffer(
            ctx,
            ptr as *mut u8,
            1,
            0,
            None,
            ptr::null_mut(),
            true,
        )
    };
    if unsafe { qjs::JS_IsException(value) } {
        release(ptr as *mut u8);
        return u32::MAX;
    }
    // SAFETY: the value is a live object.
    let class = unsafe { qjs::JS_GetClassID(value) };
    // SAFETY: `value` is the only owner besides the probe reference, which
    // is released right after.
    unsafe { qjs::JS_FreeValue(ctx, value) };
    release(ptr as *mut u8);
    // A second thread may have measured concurrently; both saw a real class
    // id, and the value cannot change between probes.
    if class != 0 && class != u32::MAX {
        CLASS.store(class as usize, Ordering::Release);
    }
    class
}

impl Runtime {
    /// Installs the reference-counted `SharedArrayBuffer` allocator.
    ///
    /// Call this before any script runs in the runtime: a block allocated
    /// before the hooks existed cannot be safely wrapped by another runtime.
    pub fn install_shared_array_buffer_hooks(&self) {
        // SAFETY: `raw_runtime` is valid for as long as this handle is alive,
        // and QuickJS copies the function table.
        unsafe {
            qjs::JS_SetSharedArrayBufferFunctions(self.raw_runtime().as_ptr(), &FUNCTIONS);
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{Context, Runtime, SharedBufferLease, Value};

    /// The transport the worker threads rely on: a lease taken from one
    /// runtime's `SharedArrayBuffer` wraps the same bytes in a second runtime
    /// on a second OS thread, and `Atomics` sees the write across runtimes.
    #[test]
    fn two_runtimes_share_one_shared_array_buffer() {
        let (lease_tx, lease_rx) = std::sync::mpsc::channel::<SharedBufferLease>();
        let reader = std::thread::spawn(move || {
            let runtime = Runtime::new().expect("reader runtime");
            runtime.install_shared_array_buffer_hooks();
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            runtime.set_interrupt_handler(Some(Box::new(move || {
                std::time::Instant::now() > deadline
            })));
            let ctx = Context::full(&runtime).expect("reader context");
            ctx.with(|ctx| {
                let sab: Value = ctx.eval("new SharedArrayBuffer(16)").expect("reader SAB");
                let lease = SharedBufferLease::new(&ctx, &sab).expect("reader lease");
                lease_tx.send(lease).expect("hand the lease to the writer");
                ctx.globals().set("sab", sab).expect("publish the reader SAB");
                ctx.eval::<(), _>(
                    "const view = new Uint8Array(sab); \
                     while (Atomics.load(view, 0) !== 1) {}",
                )
                .expect("the reader saw the writer's byte");
                let seen: u8 = ctx.eval("new Uint8Array(sab)[0]").expect("read back");
                assert_eq!(seen, 1);
            });
        });

        let lease = lease_rx.recv().expect("reader sends the lease");
        let runtime = Runtime::new().expect("writer runtime");
        runtime.install_shared_array_buffer_hooks();
        let ctx = Context::full(&runtime).expect("writer context");
        ctx.with(|ctx| {
            let buffer = lease.to_array_buffer(&ctx).expect("wrap the buffer");
            ctx.globals().set("sab", buffer).expect("publish the writer SAB");
            ctx.eval::<(), _>("Atomics.store(new Uint8Array(sab), 0, 1)")
                .expect("writer stores the byte");
        });
        reader.join().expect("reader thread");
    }
}
