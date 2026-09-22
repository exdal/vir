use std::collections::HashMap;

use ash::vk;

use crate::{Image, ImageInfo, MemoryLocation};

const IDLE_FRAMES: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ImageKey {
    extent: [u32; 3],
    format: i32,
    usage: u32,
    image_type: i32,
    mip_levels: u32,
    array_layers: u32,
    samples: u32,
    location: MemoryLocation,
}

impl From<&ImageInfo> for ImageKey {
    fn from(info: &ImageInfo) -> Self {
        Self {
            extent: [info.extent.width, info.extent.height, info.extent.depth],
            format: info.format.as_raw(),
            usage: info.usage.as_raw(),
            image_type: info.image_type.as_raw(),
            mip_levels: info.mip_levels,
            array_layers: info.array_layers,
            samples: info.samples.as_raw(),
            location: info.location,
        }
    }
}

#[derive(Debug)]
struct CachedImage {
    image: Image,
    last_use_frame: usize,
}

#[derive(Debug, Default)]
pub(super) struct FrameImageCache {
    images: HashMap<ImageKey, Vec<CachedImage>>,
    uses: HashMap<ImageKey, usize>,
}

impl FrameImageCache {
    pub(super) fn acquire(
        &mut self, info: &ImageInfo, frame: usize, create: impl FnOnce() -> Result<Image, vk::Result>,
    ) -> Result<Image, vk::Result> {
        let key = ImageKey::from(info);
        let at = *self.uses.get(&key).unwrap_or(&0);
        if let Some(entry) = self.images.get_mut(&key).and_then(|images| images.get_mut(at)) {
            entry.last_use_frame = frame;
            *self.uses.entry(key).or_default() += 1;
            return Ok(entry.image);
        }
        let image = create()?;
        self.images.entry(key).or_default().push(CachedImage {
            image,
            last_use_frame: frame,
        });
        *self.uses.entry(key).or_default() += 1;
        Ok(image)
    }

    /// Call only after the frame slot's GPU work has completed.
    pub(super) fn retire(&mut self, frame: usize) -> Vec<Image> {
        self.uses.clear();
        let mut idle = Vec::new();
        for images in self.images.values_mut() {
            images.retain(|entry| {
                if frame.saturating_sub(entry.last_use_frame) > IDLE_FRAMES {
                    idle.push(entry.image);
                    false
                } else {
                    true
                }
            });
        }
        self.images.retain(|_, images| !images.is_empty());
        idle
    }

    pub(super) fn drain(&mut self) -> Vec<Image> {
        self.uses.clear();
        self.images
            .drain()
            .flat_map(|(_, images)| images.into_iter().map(|entry| entry.image))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use ash::vk::Handle;

    use super::*;

    fn info() -> ImageInfo {
        ImageInfo::color_target(vk::Extent2D { width: 64, height: 64 }, vk::Format::R8G8B8A8_UNORM)
    }

    fn fake(id: u64, info: &ImageInfo) -> Image { Image::new(vk::Image::from_raw(id), info) }

    #[test]
    fn identical_images_are_distinct_within_a_frame_and_reused_after_retirement() {
        let info = info();
        let mut cache = FrameImageCache::default();
        let a = cache.acquire(&info, 1, || Ok(fake(1, &info))).unwrap();
        let b = cache.acquire(&info, 1, || Ok(fake(2, &info))).unwrap();
        assert_ne!(a, b);
        assert!(cache.retire(2).is_empty());
        assert_eq!(cache.acquire(&info, 2, || panic!("must reuse")).unwrap(), a);
        assert_eq!(cache.acquire(&info, 2, || panic!("must reuse")).unwrap(), b);
    }

    #[test]
    fn idle_images_are_evicted_only_after_sixteen_frames() {
        let info = info();
        let mut cache = FrameImageCache::default();
        let image = cache.acquire(&info, 1, || Ok(fake(1, &info))).unwrap();
        assert!(cache.retire(17).is_empty());
        assert_eq!(cache.retire(18), vec![image]);
        assert_eq!(
            cache
                .acquire(&info, 18, || Ok(fake(2, &info)))
                .unwrap()
                .handle()
                .as_raw(),
            2
        );
    }

    #[test]
    fn a_failed_creation_does_not_consume_an_image_ordinal() {
        let info = info();
        let mut cache = FrameImageCache::default();
        assert!(
            cache
                .acquire(&info, 1, || Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY))
                .is_err()
        );
        assert_eq!(
            cache
                .acquire(&info, 1, || Ok(fake(1, &info)))
                .unwrap()
                .handle()
                .as_raw(),
            1
        );
        assert_eq!(
            cache
                .acquire(&info, 1, || Ok(fake(2, &info)))
                .unwrap()
                .handle()
                .as_raw(),
            2
        );
    }
}
