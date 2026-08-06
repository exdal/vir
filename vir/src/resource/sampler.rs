use ash::vk;

/// What an allocator needs to hand back a sampler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SamplerInfo {
    pub mag_filter: vk::Filter,
    pub min_filter: vk::Filter,
    pub mipmap_mode: vk::SamplerMipmapMode,
    pub address_mode_u: vk::SamplerAddressMode,
    pub address_mode_v: vk::SamplerAddressMode,
    pub address_mode_w: vk::SamplerAddressMode,
}

impl Default for SamplerInfo {
    fn default() -> Self { Self::linear() }
}

impl SamplerInfo {
    pub fn linear() -> Self { Self::filtered(vk::Filter::LINEAR) }

    pub fn nearest() -> Self { Self::filtered(vk::Filter::NEAREST) }

    fn filtered(filter: vk::Filter) -> Self {
        Self {
            mag_filter: filter,
            min_filter: filter,
            mipmap_mode: vk::SamplerMipmapMode::LINEAR,
            address_mode_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            address_mode_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            address_mode_w: vk::SamplerAddressMode::CLAMP_TO_EDGE,
        }
    }

    pub fn with_mag_filter(mut self, filter: vk::Filter) -> Self {
        self.mag_filter = filter;
        self
    }

    pub fn with_min_filter(mut self, filter: vk::Filter) -> Self {
        self.min_filter = filter;
        self
    }

    /// Sets every axis at once, which is what a sampler that is not doing something exotic
    /// wants.
    pub fn with_address_mode(mut self, mode: vk::SamplerAddressMode) -> Self {
        self.address_mode_u = mode;
        self.address_mode_v = mode;
        self.address_mode_w = mode;
        self
    }
}

impl<'a> From<&SamplerInfo> for vk::SamplerCreateInfo<'a> {
    fn from(info: &SamplerInfo) -> Self {
        Self::default()
            .mag_filter(info.mag_filter)
            .min_filter(info.min_filter)
            .mipmap_mode(info.mipmap_mode)
            .address_mode_u(info.address_mode_u)
            .address_mode_v(info.address_mode_v)
            .address_mode_w(info.address_mode_w)
            .max_lod(vk::LOD_CLAMP_NONE)
    }
}
