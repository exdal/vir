#![allow(dead_code)]
pub mod image;
pub mod swapchain;

// pub struct Device {
//     handle: Arc<ash::Device>,
//     physical_device: PhysicalDevice,
//     instance: Arc<ash::Instance>,
//     swaphcain_loader: khr::swapchain::Device,
// }
//
// impl Device {
//     pub fn new(
//         handle: ash::Device, physical_device: PhysicalDevice, instance: ash::Instance,
//         swaphcain_loader: ash::khr::swapchain::Device,
//     ) -> Self {
//         Self {
//             handle: Arc::new(handle),
//             physical_device,
//             instance: Arc::new(instance),
//             swaphcain_loader,
//         }
//     }
//
//     pub fn handle(&self) -> &ash::Device { &self.handle }
//
//     pub fn physical_device(&self) -> &vk::PhysicalDevice { &self.physical_device.handle }
//
//     pub fn instance(&self) -> &ash::Instance { &self.instance }
//
//     pub fn swapchain_loader(&self) -> &khr::swapchain::Device { &self.swaphcain_loader }
// }
//
//
