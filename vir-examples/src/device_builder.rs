#![allow(dead_code)]

use std::ffi::{CStr, CString};

use ash::{khr, vk};

pub struct InstanceBuilder {
    app_name: CString,
    app_ver: u32,
    engine_name: CString,
    engine_ver: u32,
    extensions: Vec<CString>,
    min_vulkan_version: u32,
}

impl Default for InstanceBuilder {
    fn default() -> Self {
        Self {
            app_name: c"Unnamed Vulkan application".to_owned(),
            app_ver: vk::make_api_version(0, 1, 3, 0),
            engine_name: c"Unnamed Vulkan engine".to_owned(),
            engine_ver: vk::make_api_version(0, 0, 0, 0),
            extensions: Vec::default(),
            min_vulkan_version: vk::make_api_version(0, 0, 0, 0),
        }
    }
}

impl InstanceBuilder {
    pub fn require_extension(mut self, extension: CString) -> Self {
        self.extensions.push(extension);
        self
    }

    pub fn set_app_name(mut self, app_name: CString) -> Self {
        self.app_name = app_name;
        self
    }

    pub fn set_app_version(mut self, major: u32, minor: u32, patch: u32) -> Self {
        self.app_ver = vk::make_api_version(0, major, minor, patch);
        self
    }

    pub fn set_engine_name(mut self, engine_name: CString) -> Self {
        self.engine_name = engine_name;
        self
    }

    pub fn set_engine_version(mut self, major: u32, minor: u32, patch: u32) -> Self {
        self.engine_ver = vk::make_api_version(0, major, minor, patch);
        self
    }

    pub fn require_api_version(mut self, major: u32, minor: u32, patch: u32) -> Self {
        self.min_vulkan_version = vk::make_api_version(0, major, minor, patch);
        self
    }

    pub fn build(&self, entry: &ash::Entry) -> Result<ash::Instance, vk::Result> {
        let app_info = vk::ApplicationInfo::default()
            .application_name(self.app_name.as_c_str())
            .application_version(self.app_ver)
            .engine_name(self.engine_name.as_c_str())
            .engine_version(self.engine_ver)
            .api_version(self.min_vulkan_version);

        unsafe {
            entry.create_instance(
                &vk::InstanceCreateInfo::default()
                    .application_info(&app_info)
                    .enabled_extension_names(&self.extensions.iter().map(|x| x.as_ptr()).collect::<Vec<_>>()),
                None,
            )
        }
    }
}

#[derive(Debug)]
pub struct PhysicalDevice {
    pub handle: vk::PhysicalDevice,
    pub name: String,
    pub features: vk::PhysicalDeviceFeatures,
    pub properties: vk::PhysicalDeviceProperties,
    pub queue_family_properties: Vec<vk::QueueFamilyProperties>,
    pub extensions_to_enable: Vec<CString>,
}

fn get_first_queue_index(families: &[vk::QueueFamilyProperties], desired_flags: vk::QueueFlags) -> Option<u32> {
    for (i, family) in families.iter().enumerate() {
        if family.queue_flags.contains(desired_flags) {
            return Some(i as u32);
        }
    }

    None
}

fn get_separate_queue_index(
    families: &[vk::QueueFamilyProperties], desired_flags: vk::QueueFlags, undesired_flags: vk::QueueFlags,
) -> Option<u32> {
    let mut index = None;

    for (i, family) in families.iter().enumerate() {
        if family.queue_flags.contains(desired_flags) && !family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
            if !family.queue_flags.intersects(undesired_flags) {
                return Some(i as u32);
            } else {
                index = Some(i as u32);
            }
        }
    }

    index
}

fn get_dedicated_queue_index(
    families: &[vk::QueueFamilyProperties], desired_flags: vk::QueueFlags, undesired_flags: vk::QueueFlags,
) -> Option<u32> {
    for (i, family) in families.iter().enumerate() {
        if family.queue_flags.contains(desired_flags)
            && !family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
            && !family.queue_flags.intersects(undesired_flags)
        {
            return Some(i as u32);
        }
    }

    None
}

fn get_present_queue_index(
    surface_loader: &ash::khr::surface::Instance, phys_device: vk::PhysicalDevice, surface: vk::SurfaceKHR,
    families: &[vk::QueueFamilyProperties],
) -> Option<u32> {
    for (i, _family) in families.iter().enumerate() {
        if surface != vk::SurfaceKHR::null() {
            match unsafe { surface_loader.get_physical_device_surface_support(phys_device, i as u32, surface) } {
                Ok(present_support) => {
                    if present_support {
                        return Some(i as u32);
                    }
                },
                Err(_) => return None,
            }
        }
    }

    None
}

pub struct PhysicalDeviceSelector {
    min_vulkan_version: u32,
    required_extensions: Vec<CString>,
    preferred_device_type: vk::PhysicalDeviceType,
    allow_any_device_type: bool,
    require_present: bool,
    require_dedicated_compute_queue: bool,
    require_dedicated_transfer_queue: bool,
    require_separate_compute_queue: bool,
    require_separate_transfer_queue: bool,
}

impl Default for PhysicalDeviceSelector {
    fn default() -> Self {
        Self {
            min_vulkan_version: vk::make_api_version(0, 1, 3, 0),
            required_extensions: Vec::default(),
            preferred_device_type: vk::PhysicalDeviceType::DISCRETE_GPU,
            allow_any_device_type: true,
            require_present: false,
            require_dedicated_compute_queue: false,
            require_dedicated_transfer_queue: false,
            require_separate_compute_queue: false,
            require_separate_transfer_queue: false,
        }
    }
}

impl PhysicalDeviceSelector {
    pub fn set_min_api_version(mut self, major: u32, minor: u32, patch: u32) -> Self {
        self.min_vulkan_version = vk::make_api_version(0, major, minor, patch);
        self
    }

    pub fn add_required_extension(mut self, extension: CString) -> Self {
        self.required_extensions.push(extension.to_owned());
        self
    }

    pub fn set_preferred_device_type(mut self, device_type: vk::PhysicalDeviceType) -> Self {
        self.preferred_device_type = device_type;
        self
    }

    pub fn allow_any_device_type(mut self, alloy_any_type: bool) -> Self {
        self.allow_any_device_type = alloy_any_type;
        self
    }

    pub fn require_present(mut self, require: bool) -> Self {
        self.require_present = require;
        self
    }

    pub fn require_dedicated_compute_queue(mut self, require: bool) -> Self {
        self.require_dedicated_compute_queue = require;
        self
    }

    pub fn require_dedicated_transfer_queue(mut self, require: bool) -> Self {
        self.require_dedicated_transfer_queue = require;
        self
    }

    pub fn require_separate_compute_queue(mut self, require: bool) -> Self {
        self.require_separate_compute_queue = require;
        self
    }

    pub fn require_separate_transfer_queue(mut self, require: bool) -> Self {
        self.require_separate_transfer_queue = require;
        self
    }

    pub fn select(&self, instance: &ash::Instance) -> Result<PhysicalDevice, vk::Result> {
        let mut physical_devices = unsafe { instance.enumerate_physical_devices()? }
            .into_iter()
            .filter_map(move |handle| {
                let properties = unsafe { instance.get_physical_device_properties(handle) };
                if properties.api_version < self.min_vulkan_version {
                    return None;
                }

                if !self.allow_any_device_type && properties.device_type != self.preferred_device_type {
                    return None;
                }

                let queue_family_properties = unsafe { instance.get_physical_device_queue_family_properties(handle) };
                let dedicated_compute_index = get_dedicated_queue_index(
                    &queue_family_properties,
                    vk::QueueFlags::COMPUTE,
                    vk::QueueFlags::TRANSFER,
                );
                let dedicated_transfer_index = get_dedicated_queue_index(
                    &queue_family_properties,
                    vk::QueueFlags::TRANSFER,
                    vk::QueueFlags::COMPUTE,
                );
                let separate_compute_index = get_separate_queue_index(
                    &queue_family_properties,
                    vk::QueueFlags::COMPUTE,
                    vk::QueueFlags::TRANSFER,
                );
                let separate_transfer_index = get_separate_queue_index(
                    &queue_family_properties,
                    vk::QueueFlags::TRANSFER,
                    vk::QueueFlags::COMPUTE,
                );

                if self.require_dedicated_compute_queue && dedicated_compute_index.is_none() {
                    return None;
                }

                if self.require_dedicated_transfer_queue && dedicated_transfer_index.is_none() {
                    return None;
                }

                if self.require_separate_compute_queue && separate_compute_index.is_none() {
                    return None;
                }

                if self.require_separate_transfer_queue && separate_transfer_index.is_none() {
                    return None;
                }

                let extension_properties = unsafe { instance.enumerate_device_extension_properties(handle) };
                let Ok(extension_properties) = extension_properties else {
                    return None;
                };

                let extension_names = extension_properties
                    .into_iter()
                    .map(|x| x.extension_name_as_c_str().unwrap().to_owned())
                    .collect::<Vec<_>>();

                if !self.required_extensions.iter().all(|x| extension_names.contains(x)) {
                    return None;
                }

                let features = unsafe { instance.get_physical_device_features(handle) };
                let name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
                    .to_str()
                    .unwrap()
                    .to_string();

                Some(PhysicalDevice {
                    handle,
                    name,
                    features,
                    properties,
                    queue_family_properties,
                    extensions_to_enable: self.required_extensions.clone(),
                })
            })
            .collect::<Vec<_>>();

        physical_devices.sort_by_key(|physical_device| {
            let type_priority = physical_device.properties.device_type.as_raw();
            let both_same = physical_device.properties.device_type == self.preferred_device_type;
            (!both_same as u32, type_priority)
        });

        if physical_devices.is_empty() {
            return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
        }

        Ok(physical_devices.remove(0))
    }
}

#[derive(Default)]
pub struct DeviceBuilder<'a> {
    enabled_features: vk::PhysicalDeviceFeatures2<'a>,
}

impl<'a> DeviceBuilder<'a> {
    pub fn set_features(mut self, features: vk::PhysicalDeviceFeatures2<'a>) -> Self {
        self.enabled_features = features;
        self
    }

    pub fn build(
        &mut self, instance: &ash::Instance, physical_device: &PhysicalDevice,
    ) -> Result<ash::Device, vk::Result> {
        let queue_priority = [1.0_f32];
        let mut queue_create_infos = Vec::default();
        for (index, _) in physical_device.queue_family_properties.iter().enumerate() {
            let queue_create_info = vk::DeviceQueueCreateInfo::default()
                .queue_family_index(index as u32)
                .queue_priorities(&queue_priority);

            queue_create_infos.push(queue_create_info);
        }

        let extensions_to_enable = physical_device
            .extensions_to_enable
            .iter()
            .map(|x| x.as_ptr())
            .collect::<Vec<_>>();

        let create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&extensions_to_enable)
            .push_next(&mut self.enabled_features);

        unsafe { instance.create_device(physical_device.handle, &create_info, None) }
    }
}

pub struct SwapChainBuilder {
    desired_formats: Vec<vk::SurfaceFormatKHR>,
    desired_present_modes: Vec<vk::PresentModeKHR>,
    desired_width: u32,
    desired_height: u32,
    desired_min_image_count: u32,
    required_min_image_count: u32,
    old_swapchain: vk::SwapchainKHR,
    physical_device: vk::PhysicalDevice,
}

impl SwapChainBuilder {
    pub fn new(physical_device: vk::PhysicalDevice) -> Self {
        Self {
            desired_formats: Vec::default(),
            desired_present_modes: Vec::default(),
            desired_width: 256,
            desired_height: 256,
            desired_min_image_count: 0,
            required_min_image_count: 0,
            old_swapchain: vk::SwapchainKHR::default(),
            physical_device,
        }
    }

    pub fn set_old_swapchain(mut self, old_swapchain: vk::SwapchainKHR) -> Self {
        self.old_swapchain = old_swapchain;
        self
    }

    pub fn set_desired_extent(mut self, width: u32, height: u32) -> Self {
        self.desired_width = width;
        self.desired_height = height;
        self
    }

    pub fn set_desired_format(mut self, surface_format: vk::SurfaceFormatKHR) -> Self {
        self.desired_formats.insert(0, surface_format);
        self
    }

    pub fn set_desired_present_mode(mut self, present_mode: vk::PresentModeKHR) -> Self {
        self.desired_present_modes.insert(0, present_mode);
        self
    }

    pub fn set_desired_min_image_count(mut self, count: u32) -> Self {
        self.desired_min_image_count = count;
        self
    }

    pub fn set_required_min_image_count(mut self, count: u32) -> Self {
        self.required_min_image_count = count;
        self
    }

    pub fn build(
        &mut self, surface_loader: &ash::khr::surface::Instance, surface: &vk::SurfaceKHR,
        swapchain_loader: &khr::swapchain::Device,
    ) -> Result<(vk::SwapchainKHR, vk::Format, vk::Extent2D), vk::Result> {
        if self.desired_formats.is_empty() {
            let rgba_srgb_format = vk::SurfaceFormatKHR::default()
                .format(vk::Format::R8G8B8A8_SRGB)
                .color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR);
            self.desired_formats.push(rgba_srgb_format);

            let bgra_srgb_format = vk::SurfaceFormatKHR::default()
                .format(vk::Format::B8G8R8A8_SRGB)
                .color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR);
            self.desired_formats.push(bgra_srgb_format);
        }

        if self.desired_present_modes.is_empty() {
            self.desired_present_modes.push(vk::PresentModeKHR::FIFO);
        }

        let capabilities =
            unsafe { surface_loader.get_physical_device_surface_capabilities(self.physical_device, *surface)? };
        let surface_formats =
            unsafe { surface_loader.get_physical_device_surface_formats(self.physical_device, *surface)? };
        let present_modes =
            unsafe { surface_loader.get_physical_device_surface_present_modes(self.physical_device, *surface)? };

        let mut image_count;
        if self.required_min_image_count >= 1 {
            if self.required_min_image_count < capabilities.min_image_count {
                return Err(vk::Result::ERROR_SURFACE_LOST_KHR);
            }

            image_count = self.required_min_image_count;
        } else if self.desired_min_image_count == 0 {
            image_count = capabilities.min_image_count + 1;
        } else {
            image_count = self.desired_min_image_count;
            image_count = image_count.max(capabilities.min_image_count);
        }

        if capabilities.max_image_count > 0 && image_count > capabilities.max_image_count {
            image_count = capabilities.max_image_count;
        }

        let extent = if capabilities.current_extent.width != u32::MAX {
            capabilities.current_extent
        } else {
            vk::Extent2D {
                width: self
                    .desired_width
                    .max(capabilities.min_image_extent.width)
                    .min(capabilities.max_image_extent.width),
                height: self
                    .desired_height
                    .max(capabilities.min_image_extent.height)
                    .min(capabilities.max_image_extent.height),
            }
        };

        let surface_format = self
            .desired_formats
            .iter()
            .copied()
            .find(|desired| {
                surface_formats
                    .iter()
                    .any(|available| desired.format == available.format && desired.color_space == available.color_space)
            })
            .or_else(|| surface_formats.first().copied())
            .ok_or(vk::Result::ERROR_FORMAT_NOT_SUPPORTED)?;

        let present_mode = self
            .desired_present_modes
            .iter()
            .copied()
            .find(|desired| present_modes.iter().any(|available| desired == available))
            .or(Some(vk::PresentModeKHR::FIFO))
            .ok_or(vk::Result::ERROR_SURFACE_LOST_KHR)?;

        let create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(*surface)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_extent(extent)
            .image_array_layers(1)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .present_mode(present_mode)
            .pre_transform(capabilities.current_transform)
            .clipped(true)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .old_swapchain(self.old_swapchain);

        let handle = unsafe { swapchain_loader.create_swapchain(&create_info, None) }?;

        Ok((handle, surface_format.format, extent))
    }
}
