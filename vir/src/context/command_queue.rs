use std::ptr::NonNull;

use ash::vk;
use bitflags::bitflags;

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct DomainFlag: u32 {
        const None = 0;
        const Host = 1 << 0;
        const Present = 1 << 1;
        const Graphics = 1 << 2;
        const Compute = 1 << 3;
        const Transfer = 1 << 4;
    }
}

impl Default for DomainFlag {
    fn default() -> Self { Self::None }
}

pub struct CommandQueue {
    inner: vk::Queue,
    device: NonNull<ash::Device>,
    family_index: u32,
    domain_flags: DomainFlag,
    timeline: vk::Semaphore,
}

impl CommandQueue {
    pub fn new(
        inner: vk::Queue, device: NonNull<ash::Device>, family_index: u32, domain_flags: DomainFlag,
        timeline: vk::Semaphore,
    ) -> Self {
        Self {
            inner,
            device,
            family_index,
            domain_flags,
            timeline,
        }
    }

    pub fn inner(&self) -> vk::Queue { self.inner }

    pub fn family_index(&self) -> u32 { self.family_index }

    pub fn domain_flags(&self) -> DomainFlag { self.domain_flags }

    pub fn semaphore(&self) -> &vk::Semaphore { &self.timeline }

    pub fn submit(&self, submits: &[vk::SubmitInfo2]) -> Result<(), vk::Result> {
        unsafe {
            self.device
                .as_ref()
                .queue_submit2(self.inner, submits, vk::Fence::null())
        }
    }
}
