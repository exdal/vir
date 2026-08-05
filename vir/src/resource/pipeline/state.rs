use std::hash::{Hash, Hasher};

use ash::vk;
use bitflags::bitflags;

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct DynamicStateFlags: u32 {
        const None = 0;
        const Viewport = 1 << 0;
        const Scissor = 1 << 1;
    }
}

impl Default for DynamicStateFlags {
    fn default() -> Self { Self::Viewport | Self::Scissor }
}

#[derive(Debug, Clone, Copy)]
pub enum Rect2D {
    Framebuffer,
    Absolute(vk::Rect2D),
    Relative { x: f32, y: f32, width: f32, height: f32 },
}

impl Default for Rect2D {
    fn default() -> Self { Self::Framebuffer }
}

impl Rect2D {
    pub const fn framebuffer() -> Self { Self::Framebuffer }

    pub const fn absolute(x: i32, y: i32, width: u32, height: u32) -> Self {
        Self::Absolute(vk::Rect2D {
            offset: vk::Offset2D { x, y },
            extent: vk::Extent2D { width, height },
        })
    }

    pub const fn relative(x: f32, y: f32, width: f32, height: f32) -> Self { Self::Relative { x, y, width, height } }

    pub fn resolve(&self, area: vk::Rect2D) -> vk::Rect2D {
        match *self {
            Self::Framebuffer => area,
            Self::Absolute(rect) => rect,
            Self::Relative { x, y, width, height } => vk::Rect2D {
                offset: vk::Offset2D {
                    x: area.offset.x + (area.extent.width as f32 * x) as i32,
                    y: area.offset.y + (area.extent.height as f32 * y) as i32,
                },
                extent: vk::Extent2D {
                    width: (area.extent.width as f32 * width) as u32,
                    height: (area.extent.height as f32 * height) as u32,
                },
            },
        }
    }

    // floats are not Eq/Hash, so keying goes through the raw bits
    fn key(&self) -> (u8, [u32; 4]) {
        match *self {
            Self::Framebuffer => (0, [0; 4]),
            Self::Absolute(rect) => (
                1,
                [
                    rect.offset.x as u32,
                    rect.offset.y as u32,
                    rect.extent.width,
                    rect.extent.height,
                ],
            ),
            Self::Relative { x, y, width, height } => {
                (2, [x.to_bits(), y.to_bits(), width.to_bits(), height.to_bits()])
            },
        }
    }
}

impl PartialEq for Rect2D {
    fn eq(&self, other: &Self) -> bool { self.key() == other.key() }
}

impl Eq for Rect2D {}

impl Hash for Rect2D {
    fn hash<H: Hasher>(&self, state: &mut H) { self.key().hash(state); }
}

#[derive(Debug, Clone, Copy)]
pub struct Viewport {
    pub rect: Rect2D,
    pub min_depth: f32,
    pub max_depth: f32,
}

impl Default for Viewport {
    fn default() -> Self {
        Self {
            rect: Rect2D::Framebuffer,
            min_depth: 0.0,
            max_depth: 1.0,
        }
    }
}

impl From<Rect2D> for Viewport {
    fn from(rect: Rect2D) -> Self {
        Self {
            rect,
            ..Self::default()
        }
    }
}

impl Viewport {
    pub fn framebuffer() -> Self { Self::default() }

    pub fn with_depth_range(mut self, min_depth: f32, max_depth: f32) -> Self {
        self.min_depth = min_depth;
        self.max_depth = max_depth;
        self
    }

    pub fn resolve(&self, area: vk::Rect2D) -> ResolvedViewport {
        let rect = self.rect.resolve(area);
        ResolvedViewport {
            x: rect.offset.x as f32,
            y: rect.offset.y as f32,
            width: rect.extent.width as f32,
            height: rect.extent.height as f32,
            min_depth: self.min_depth,
            max_depth: self.max_depth,
        }
    }

    fn key(&self) -> ((u8, [u32; 4]), u32, u32) {
        (self.rect.key(), self.min_depth.to_bits(), self.max_depth.to_bits())
    }
}

impl PartialEq for Viewport {
    fn eq(&self, other: &Self) -> bool { self.key() == other.key() }
}

impl Eq for Viewport {}

impl Hash for Viewport {
    fn hash<H: Hasher>(&self, state: &mut H) { self.key().hash(state); }
}

/// A [`Viewport`] with the render area substituted in.
#[derive(Debug, Clone, Copy, Default)]
pub struct ResolvedViewport {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub min_depth: f32,
    pub max_depth: f32,
}

impl ResolvedViewport {
    fn key(&self) -> [u32; 6] {
        [
            self.x.to_bits(),
            self.y.to_bits(),
            self.width.to_bits(),
            self.height.to_bits(),
            self.min_depth.to_bits(),
            self.max_depth.to_bits(),
        ]
    }
}

impl PartialEq for ResolvedViewport {
    fn eq(&self, other: &Self) -> bool { self.key() == other.key() }
}

impl Eq for ResolvedViewport {}

impl Hash for ResolvedViewport {
    fn hash<H: Hasher>(&self, state: &mut H) { self.key().hash(state); }
}

impl From<ResolvedViewport> for vk::Viewport {
    fn from(viewport: ResolvedViewport) -> Self {
        Self {
            x: viewport.x,
            y: viewport.y,
            width: viewport.width,
            height: viewport.height,
            min_depth: viewport.min_depth,
            max_depth: viewport.max_depth,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RenderingState {
    pub color_formats: Vec<vk::Format>,
    pub samples: vk::SampleCountFlags,
}

impl Default for RenderingState {
    fn default() -> Self {
        Self {
            color_formats: Vec::new(),
            samples: vk::SampleCountFlags::TYPE_1,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RasterizationState {
    pub polygon_mode: vk::PolygonMode,
    pub cull_mode: vk::CullModeFlags,
    pub front_face: vk::FrontFace,
    pub line_width: f32,
    pub depth_clamp_enable: bool,
    pub rasterizer_discard_enable: bool,
}

impl Default for RasterizationState {
    fn default() -> Self {
        Self {
            polygon_mode: vk::PolygonMode::FILL,
            cull_mode: vk::CullModeFlags::NONE,
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            line_width: 1.0,
            depth_clamp_enable: false,
            rasterizer_discard_enable: false,
        }
    }
}

impl RasterizationState {
    fn key(&self) -> (i32, u32, i32, u32, bool, bool) {
        (
            self.polygon_mode.as_raw(),
            self.cull_mode.as_raw(),
            self.front_face.as_raw(),
            self.line_width.to_bits(),
            self.depth_clamp_enable,
            self.rasterizer_discard_enable,
        )
    }
}

impl PartialEq for RasterizationState {
    fn eq(&self, other: &Self) -> bool { self.key() == other.key() }
}

impl Eq for RasterizationState {}

impl Hash for RasterizationState {
    fn hash<H: Hasher>(&self, state: &mut H) { self.key().hash(state); }
}

impl<'a> From<RasterizationState> for vk::PipelineRasterizationStateCreateInfo<'a> {
    fn from(state: RasterizationState) -> Self {
        Self::default()
            .polygon_mode(state.polygon_mode)
            .cull_mode(state.cull_mode)
            .front_face(state.front_face)
            .line_width(state.line_width)
            .depth_clamp_enable(state.depth_clamp_enable)
            .rasterizer_discard_enable(state.rasterizer_discard_enable)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ColorBlendAttachmentState {
    pub blend_enable: bool,
    pub src_color_blend_factor: vk::BlendFactor,
    pub dst_color_blend_factor: vk::BlendFactor,
    pub color_blend_op: vk::BlendOp,
    pub src_alpha_blend_factor: vk::BlendFactor,
    pub dst_alpha_blend_factor: vk::BlendFactor,
    pub alpha_blend_op: vk::BlendOp,
    pub color_write_mask: vk::ColorComponentFlags,
}

impl Default for ColorBlendAttachmentState {
    fn default() -> Self { BlendPreset::Off.into() }
}

impl From<ColorBlendAttachmentState> for vk::PipelineColorBlendAttachmentState {
    fn from(state: ColorBlendAttachmentState) -> Self {
        Self::default()
            .blend_enable(state.blend_enable)
            .src_color_blend_factor(state.src_color_blend_factor)
            .dst_color_blend_factor(state.dst_color_blend_factor)
            .color_blend_op(state.color_blend_op)
            .src_alpha_blend_factor(state.src_alpha_blend_factor)
            .dst_alpha_blend_factor(state.dst_alpha_blend_factor)
            .alpha_blend_op(state.alpha_blend_op)
            .color_write_mask(state.color_write_mask)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlendPreset {
    Off,
    AlphaBlend,
    PremultipliedAlphaBlend,
    Additive,
}

impl From<BlendPreset> for ColorBlendAttachmentState {
    fn from(preset: BlendPreset) -> Self {
        let base = ColorBlendAttachmentState {
            blend_enable: preset != BlendPreset::Off,
            src_color_blend_factor: vk::BlendFactor::ONE,
            dst_color_blend_factor: vk::BlendFactor::ZERO,
            color_blend_op: vk::BlendOp::ADD,
            src_alpha_blend_factor: vk::BlendFactor::ONE,
            dst_alpha_blend_factor: vk::BlendFactor::ZERO,
            alpha_blend_op: vk::BlendOp::ADD,
            color_write_mask: vk::ColorComponentFlags::RGBA,
        };

        match preset {
            BlendPreset::Off => base,
            BlendPreset::AlphaBlend => ColorBlendAttachmentState {
                src_color_blend_factor: vk::BlendFactor::SRC_ALPHA,
                dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                src_alpha_blend_factor: vk::BlendFactor::ONE,
                dst_alpha_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                ..base
            },
            BlendPreset::PremultipliedAlphaBlend => ColorBlendAttachmentState {
                src_color_blend_factor: vk::BlendFactor::ONE,
                dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                src_alpha_blend_factor: vk::BlendFactor::ONE,
                dst_alpha_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                ..base
            },
            BlendPreset::Additive => ColorBlendAttachmentState {
                dst_color_blend_factor: vk::BlendFactor::ONE,
                dst_alpha_blend_factor: vk::BlendFactor::ONE,
                ..base
            },
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct PushConstants {
    pub offset: u32,
    pub data: Vec<u8>,
}

impl PushConstants {
    pub fn is_empty(&self) -> bool { self.data.is_empty() }

    pub fn size(&self) -> u32 { self.data.len() as u32 }

    pub fn end(&self) -> u32 { self.offset + self.size() }

    pub fn write(&mut self, offset: u32, bytes: &[u8]) {
        if self.is_empty() {
            self.offset = offset;
            self.data = bytes.to_vec();
            return;
        }

        let start = self.offset.min(offset);
        let end = self.end().max(offset + bytes.len() as u32);

        if (start, end) != (self.offset, self.end()) {
            let mut widened = vec![0; (end - start) as usize];
            let head = (self.offset - start) as usize;
            widened[head..head + self.data.len()].copy_from_slice(&self.data);
            self.offset = start;
            self.data = widened;
        }

        let head = (offset - self.offset) as usize;
        self.data[head..head + bytes.len()].copy_from_slice(bytes);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StateChange {
    PrimitiveTopology(vk::PrimitiveTopology),
    Rasterization(RasterizationState),
    DynamicState(DynamicStateFlags),
    Viewport {
        index: u32,
        viewport: Viewport,
    },
    Scissor {
        index: u32,
        rect: Rect2D,
    },
    ColorBlend {
        index: Option<u32>,
        blend: ColorBlendAttachmentState,
    },
    PushConstants {
        offset: u32,
        data: Vec<u8>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PassState {
    pub rendering: RenderingState,
    pub topology: vk::PrimitiveTopology,
    pub rasterization: RasterizationState,
    pub blend: Vec<ColorBlendAttachmentState>,
    pub viewports: Vec<Viewport>,
    pub scissors: Vec<Rect2D>,
    pub dynamic: DynamicStateFlags,
    pub push_constants: PushConstants,
}

impl Default for PassState {
    fn default() -> Self {
        Self {
            rendering: RenderingState::default(),
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            rasterization: RasterizationState::default(),
            blend: Vec::new(),
            viewports: vec![Viewport::default()],
            scissors: vec![Rect2D::default()],
            dynamic: DynamicStateFlags::default(),
            push_constants: PushConstants::default(),
        }
    }
}

impl PassState {
    pub fn for_rendering(rendering: RenderingState) -> Self {
        let blend = vec![ColorBlendAttachmentState::default(); rendering.color_formats.len()];
        Self {
            rendering,
            blend,
            ..Self::default()
        }
    }

    pub fn apply(&mut self, change: StateChange) {
        fn slot<T: Default + Clone>(slots: &mut Vec<T>, index: u32) -> &mut T {
            let index = index as usize;
            if index >= slots.len() {
                slots.resize(index + 1, T::default());
            }
            &mut slots[index]
        }

        match change {
            StateChange::PrimitiveTopology(topology) => self.topology = topology,
            StateChange::Rasterization(rasterization) => self.rasterization = rasterization,
            StateChange::DynamicState(dynamic) => self.dynamic = dynamic,
            StateChange::Viewport { index, viewport } => *slot(&mut self.viewports, index) = viewport,
            StateChange::Scissor { index, rect } => *slot(&mut self.scissors, index) = rect,
            StateChange::ColorBlend {
                index: Some(index),
                blend,
            } => *slot(&mut self.blend, index) = blend,
            StateChange::ColorBlend { index: None, blend } => {
                if self.blend.is_empty() {
                    self.blend.push(blend);
                } else {
                    self.blend.iter_mut().for_each(|attachment| *attachment = blend);
                }
            },
            StateChange::PushConstants { offset, data } => self.push_constants.write(offset, &data),
        }
    }

    pub fn resolve(&self, area: vk::Rect2D) -> (PipelineState, DynamicValues) {
        let count = self.viewports.len().max(self.scissors.len()).max(1);
        let viewport_at = |index: usize| self.viewports.get(index).copied().unwrap_or_default();
        let scissor_at = |index: usize| self.scissors.get(index).copied().unwrap_or_default();

        let blend = (0..self.rendering.color_formats.len())
            .map(|index| self.blend.get(index).copied().unwrap_or_default())
            .collect();

        let mut state = PipelineState {
            rendering: self.rendering.clone(),
            topology: self.topology,
            rasterization: self.rasterization,
            blend,
            viewport_count: count as u32,
            viewports: Vec::new(),
            scissors: Vec::new(),
            dynamic: self.dynamic,
        };
        let mut values = DynamicValues {
            push_constants: self.push_constants.clone(),
            ..Default::default()
        };

        if self.dynamic.contains(DynamicStateFlags::Viewport) {
            values.viewports = (0..count).map(viewport_at).collect();
        } else {
            state.viewports = (0..count).map(|index| viewport_at(index).resolve(area)).collect();
        }

        if self.dynamic.contains(DynamicStateFlags::Scissor) {
            values.scissors = (0..count).map(scissor_at).collect();
        } else {
            state.scissors = (0..count).map(|index| scissor_at(index).resolve(area)).collect();
        }

        (state, values)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PipelineState {
    pub rendering: RenderingState,
    pub topology: vk::PrimitiveTopology,
    pub rasterization: RasterizationState,
    pub blend: Vec<ColorBlendAttachmentState>,
    pub viewport_count: u32,
    pub viewports: Vec<ResolvedViewport>,
    pub scissors: Vec<vk::Rect2D>,
    pub dynamic: DynamicStateFlags,
}

impl Default for PipelineState {
    fn default() -> Self { PassState::default().resolve(vk::Rect2D::default()).0 }
}

impl PipelineState {
    pub fn dynamic_states(&self) -> Vec<vk::DynamicState> {
        let mut states = Vec::new();
        if self.dynamic.contains(DynamicStateFlags::Viewport) {
            states.push(vk::DynamicState::VIEWPORT);
        }
        if self.dynamic.contains(DynamicStateFlags::Scissor) {
            states.push(vk::DynamicState::SCISSOR);
        }
        states
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct DynamicValues {
    pub viewports: Vec<Viewport>,
    pub scissors: Vec<Rect2D>,
    pub push_constants: PushConstants,
}

impl DynamicValues {
    pub fn viewports(&self, area: vk::Rect2D) -> Vec<ResolvedViewport> {
        self.viewports.iter().map(|viewport| viewport.resolve(area)).collect()
    }

    pub fn scissors(&self, area: vk::Rect2D) -> Vec<vk::Rect2D> {
        self.scissors.iter().map(|rect| rect.resolve(area)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_first_write_starts_the_run_where_it_lands() {
        let mut constants = PushConstants::default();
        constants.write(16, &[1, 2, 3, 4]);
        assert_eq!(constants.offset, 16);
        assert_eq!(constants.data, vec![1, 2, 3, 4]);
    }

    #[test]
    fn a_later_write_overwrites_the_bytes_it_lands_on() {
        let mut constants = PushConstants::default();
        constants.write(0, &[1, 2, 3, 4]);
        constants.write(2, &[9, 9]);
        assert_eq!(constants.offset, 0);
        assert_eq!(constants.data, vec![1, 2, 9, 9]);
    }

    #[test]
    fn writes_on_both_sides_widen_the_run_and_zero_the_gap() {
        let mut constants = PushConstants::default();
        constants.write(4, &[1, 2]);
        constants.write(10, &[3]);
        constants.write(0, &[7, 7]);

        assert_eq!(constants.offset, 0);
        assert_eq!(constants.data, vec![7, 7, 0, 0, 1, 2, 0, 0, 0, 0, 3]);
        assert_eq!(constants.end(), 11);
    }

    #[test]
    fn pushed_bytes_stay_out_of_the_permutation_key() {
        let mut with_constants = PassState::default();
        with_constants.apply(StateChange::PushConstants {
            offset: 0,
            data: vec![1, 2, 3, 4],
        });

        let area = vk::Rect2D::default();
        let (plain_state, plain_values) = PassState::default().resolve(area);
        let (state, values) = with_constants.resolve(area);

        assert_eq!(state, plain_state);
        assert!(plain_values.push_constants.is_empty());
        assert_eq!(values.push_constants.data, vec![1, 2, 3, 4]);
    }
}
