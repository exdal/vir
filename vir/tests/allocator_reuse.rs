use ash::vk;
use vir::{BufferInfo, Context, ImageInfo, MemoryLocation, allocator::Allocator};

#[test]
fn vulkan_buffers_and_images_reuse_backing_resources() {
    let Ok(entry) = (unsafe { ash::Entry::load() }) else {
        eprintln!("Vulkan loader is unavailable; skipping allocator integration test");
        return;
    };
    let app_info = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_2);
    let instance_info = vk::InstanceCreateInfo::default().application_info(&app_info);
    let Ok(instance) = (unsafe { entry.create_instance(&instance_info, None) }) else {
        eprintln!("Vulkan instance is unavailable; skipping allocator integration test");
        return;
    };
    let physical_devices = unsafe { instance.enumerate_physical_devices() }.unwrap();
    let Some((physical_device, family)) = physical_devices.iter().find_map(|&physical_device| {
        let families = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        families
            .iter()
            .position(|family| family.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .map(|at| (physical_device, at as u32))
    }) else {
        unsafe { instance.destroy_instance(None) };
        eprintln!("Vulkan device is unavailable; skipping allocator integration test");
        return;
    };

    let mut supported12 = vk::PhysicalDeviceVulkan12Features::default();
    let mut supported = vk::PhysicalDeviceFeatures2::default().push_next(&mut supported12);
    unsafe { instance.get_physical_device_features2(physical_device, &mut supported) };
    if supported12.buffer_device_address != vk::TRUE {
        unsafe { instance.destroy_instance(None) };
        eprintln!("Buffer device address is unavailable; skipping allocator integration test");
        return;
    }

    let priorities = [1.0];
    let queue_info = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(family)
        .queue_priorities(&priorities)];
    let mut enabled12 = vk::PhysicalDeviceVulkan12Features::default().buffer_device_address(true);
    let device_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_info)
        .push_next(&mut enabled12);
    let device = unsafe { instance.create_device(physical_device, &device_info, None) }.unwrap();
    let destroy_device = device.clone();
    let destroy_instance = instance.clone();
    let ctx = Context::new(device, physical_device, instance, &entry).unwrap();

    let info = BufferInfo::new(1024, vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::CpuToGpu);
    let mut persistent = ctx.create_persistent_allocator();
    let a = persistent.allocate_buffer(&info).unwrap();
    let b = persistent.allocate_buffer(&info).unwrap();
    assert_eq!(a.handle(), b.handle());
    assert_ne!(a.offset(), b.offset());
    persistent.deallocate_buffer(a);
    let c = persistent.allocate_buffer(&info).unwrap();
    assert_eq!(c.handle(), b.handle());
    assert_eq!(c.offset(), a.offset());
    assert_ne!(c, a);
    persistent.deallocate_buffer(b);
    persistent.deallocate_buffer(c);

    let mut superframe = ctx.create_super_frame_allocator(1);
    let image_info = ImageInfo::color_target(vk::Extent2D { width: 16, height: 16 }, vk::Format::R8G8B8A8_UNORM);
    let frame = superframe.get_next_frame().unwrap();
    let frame_a = frame.allocate_buffer(&info).unwrap();
    let frame_b = frame.allocate_buffer(&info).unwrap();
    assert_eq!(frame_a.handle(), frame_b.handle());
    assert_ne!(frame_a.offset(), frame_b.offset());
    let image_a = frame.allocate_image(&image_info).unwrap();
    let image_b = frame.allocate_image(&image_info).unwrap();
    assert_ne!(image_a, image_b);

    let frame = superframe.get_next_frame().unwrap();
    let frame_again = frame.allocate_buffer(&info).unwrap();
    assert_eq!(frame_again.handle(), frame_a.handle());
    assert_eq!(frame_again.offset(), frame_a.offset());
    assert_eq!(frame.allocate_image(&image_info).unwrap(), image_a);
    assert_eq!(frame.allocate_image(&image_info).unwrap(), image_b);

    drop(superframe);
    drop(persistent);
    drop(ctx);
    unsafe {
        destroy_device.destroy_device(None);
        destroy_instance.destroy_instance(None);
    }
}
