use ash::{vk, vk::Handle};
use vir::{Context, DomainFlag};

use crate::error::{Failure, guard_result, guard_void};

#[repr(C)]
pub struct VirContextCreateInfo {
    pub get_instance_proc_addr: Option<vk::PFN_vkGetInstanceProcAddr>,
    pub instance: usize,
    pub physical_device: usize,
    pub device: usize,
}

pub struct VirContext {
    pub ctx: Context,
    domains: Vec<DomainFlag>,
    _entry: ash::Entry,
}

impl Drop for VirContext {
    fn drop(&mut self) {
        if let Err(err) = unsafe { self.ctx.device().device_wait_idle() } {
            eprintln!("vir-ffi: device_wait_idle failed during teardown: {err}");
        }

        for &domain in &self.domains {
            if let Some(queue) = self.ctx.command_queue_by_domain(domain) {
                unsafe { self.ctx.device().destroy_semaphore(*queue.semaphore(), None) };
            }
        }
    }
}

pub(crate) fn context<'a>(c: *mut VirContext) -> Result<&'a mut VirContext, Failure> {
    unsafe { c.as_mut() }.ok_or_else(|| Failure::new("context is null"))
}

pub(crate) fn out_ptr<'a, T>(out: *mut T) -> Result<&'a mut T, Failure> {
    unsafe { out.as_mut() }.ok_or_else(|| Failure::new("out pointer is null"))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_context_create(
    info: *const VirContextCreateInfo, out: *mut *mut VirContext,
) -> vk::Result {
    guard_result(|| {
        let info = unsafe { info.as_ref() }.ok_or_else(|| Failure::new("create info is null"))?;
        let out = out_ptr(out)?;
        let get_instance_proc_addr = info
            .get_instance_proc_addr
            .ok_or_else(|| Failure::new("get_instance_proc_addr is null"))?;
        if info.instance == 0 || info.physical_device == 0 || info.device == 0 {
            return Err(Failure::new("instance, physical_device and device must not be null"));
        }

        let entry = unsafe { ash::Entry::from_static_fn(ash::StaticFn { get_instance_proc_addr }) };
        let instance = unsafe { ash::Instance::load(entry.static_fn(), vk::Instance::from_raw(info.instance as _)) };
        let device = unsafe { ash::Device::load(instance.fp_v1_0(), vk::Device::from_raw(info.device as _)) };
        let physical_device = vk::PhysicalDevice::from_raw(info.physical_device as _);
        let ctx = Context::new(device, physical_device, instance, &entry)?;

        *out = Box::into_raw(Box::new(VirContext {
            ctx,
            domains: Vec::new(),
            _entry: entry,
        }));

        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_context_destroy(c: *mut VirContext) {
    guard_void(|| {
        if !c.is_null() {
            drop(unsafe { Box::from_raw(c) });
        }

        Ok(())
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_context_create_command_queue(
    c: *mut VirContext, queue_family_index: u32, domain: u32,
) -> vk::Result {
    guard_result(|| {
        let context = context(c)?;
        let domain = DomainFlag::from_bits_retain(domain);
        if context.domains.contains(&domain) {
            return Err(Failure::new(format!("a {domain:?} queue already exists")));
        }

        context.ctx.create_command_queue(queue_family_index, domain);
        context.domains.push(domain);

        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_context_wait_idle(c: *mut VirContext) -> vk::Result {
    guard_result(|| {
        unsafe { context(c)?.ctx.device().device_wait_idle() }?;
        Ok(())
    })
}
