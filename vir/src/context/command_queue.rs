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

pub struct CommandQueue {
    inner: vk::Queue,
    family_index: u32,
    domain_flags: DomainFlag,
    timeline: vk::Semaphore,
}

impl CommandQueue {
    pub fn new(inner: vk::Queue, family_index: u32, domain_flags: DomainFlag, timeline: vk::Semaphore) -> Self {
        Self {
            inner,
            family_index,
            domain_flags,
            timeline,
        }
    }

    pub fn family_index(&self) -> u32 { self.family_index }

    pub fn semaphore(&self) -> &vk::Semaphore { &self.timeline }
}
