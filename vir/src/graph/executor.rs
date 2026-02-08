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
