//! Every scalar descriptor type supported by `vir`, without requiring a ray-tracing-capable GPU.
//!
//! The companion Slang module declares and accesses all ten Vulkan descriptor types. This
//! executable reflects those real SPIR-V entry points, records the matching host-side payloads,
//! verifies that every generic binding is compatible, and prints the resulting table and IR.
//!
//! There is deliberately no draw or dispatch here: acceleration structures and input attachments
//! need optional Vulkan facilities to execute. The non-null raw handles below are illustrative
//! placeholders and are never submitted to Vulkan. In an executing graph, they must be live
//! caller-owned objects backed by the resources passed alongside them.

use std::collections::BTreeMap;

use ash::vk::{self, Handle};
use vir::{
    BufferInfo,
    Descriptor,
    DescriptorBinding,
    IR,
    ImageInfo,
    MemoryLocation,
    Module,
    Program,
    Unchecked,
    resource::shader,
};
use vir_examples::read_spirv;

const COMPUTE_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/descriptors.comp.spv"));
const INPUT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/descriptors.input.frag.spv"));

const METHODS: [&str; 10] = [
    "bind_sampler",
    "bind_image",
    "bind_texture",
    "bind_image",
    "bind_texel_buffer",
    "bind_texel_buffer",
    "bind_buffer",
    "bind_buffer_range",
    "bind_acceleration_structure",
    "bind_image",
];

fn reflected_bindings() -> Result<Vec<DescriptorBinding>, vk::Result> {
    let mut bindings = Vec::new();
    for spirv in [COMPUTE_SPV, INPUT_SPV] {
        bindings.extend(shader::reflect(&read_spirv(spirv))?.bindings);
    }
    bindings.sort_unstable_by_key(|binding| (binding.set, binding.binding));
    Ok(bindings)
}

/// Records each payload shape once. Reflection supplies the distinctions intentionally absent
/// from these calls: sampled/storage/input image, uniform/storage texel buffer, and
/// uniform/storage buffer.
fn descriptor_payload_program() -> Result<Program, vk::Result> {
    let mut module = Module::default();
    let extent = vk::Extent2D::default().width(1).height(1);
    let image = |name| ImageInfo::color_target(extent, vk::Format::R8G8B8A8_UNORM).with_name(name);

    let sampled_image = module.transient_image(&image("sampled image"));
    let combined_image = module.transient_image(&image("combined image sampler"));
    let storage_image = module.transient_image(&image("storage image"));
    let input_attachment = module.transient_image(&image("input attachment"));
    let buffer = module.transient_buffer(
        &BufferInfo::new(256, vk::BufferUsageFlags::empty(), MemoryLocation::GpuOnly)
            .with_name("descriptor backing buffer"),
    );

    let sampler = vk::Sampler::from_raw(1);
    let uniform_view = vk::BufferView::from_raw(2);
    let storage_view = vk::BufferView::from_raw(3);
    let acceleration_structure = vk::AccelerationStructureKHR::from_raw(4);

    let end = module
        .begin_compute([])
        .with_name("descriptor payloads")
        // VK_DESCRIPTOR_TYPE_SAMPLER
        .bind_sampler(0, 0, sampler)
        // VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE
        .bind_image(0, 1, sampled_image)
        // VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
        .bind_texture(0, 2, combined_image, sampler)
        // VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: same image payload API as sampled images.
        .bind_image(0, 3, storage_image)
        // VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER
        .bind_texel_buffer(0, 4, buffer, uniform_view)
        // VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER: reflection distinguishes the same payload.
        .bind_texel_buffer(0, 5, buffer, storage_view)
        // VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
        .bind_buffer(0, 6, buffer)
        // VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: a range is optional for either buffer type.
        .bind_buffer_range(0, 7, buffer, 16, 64)
        // VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR. The buffer lets the graph synchronize
        // the memory behind the caller-owned acceleration-structure handle.
        .bind_acceleration_structure(0, 8, buffer, acceleration_structure)
        // VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT: the third use of the generic image payload.
        .bind_image(0, 9, input_attachment)
        .end_compute();

    // With no command consuming the writes, compiling needs no Vulkan pipeline object. The table
    // printed by `main` pairs these payloads with the real types reflected above.
    module.compile(&Unchecked, end)
}

fn written_descriptors(program: &Program) -> BTreeMap<(u32, u32), Descriptor> {
    program
        .instructions()
        .iter()
        .filter_map(|(_, instruction)| match instruction {
            IR::WriteDescriptor {
                set,
                binding,
                descriptor,
                ..
            } => Some(((*set, *binding), *descriptor)),
            _ => None,
        })
        .collect()
}

fn main() -> Result<(), vk::Result> {
    let bindings = reflected_bindings()?;
    let program = descriptor_payload_program()?;
    let written = written_descriptors(&program);

    println!("set  binding  reflected Vulkan type                 reflected access                  host binding");
    println!(
        "---  -------  ------------------------------------  --------------------------------  \
         ---------------------------"
    );
    for binding in &bindings {
        let descriptor = written
            .get(&(binding.set, binding.binding))
            .expect("the example should bind every reflected descriptor");
        assert!(descriptor.supports_type(binding.descriptor_type));
        let descriptor_type = format!("{:?}", binding.descriptor_type);
        let access = format!("{:?}", binding.access);
        println!(
            "{:>3}  {:>7}  {descriptor_type:<36}  {access:<32}  {}",
            binding.set, binding.binding, METHODS[binding.binding as usize],
        );
    }

    println!("\nPayload-only IR (types are inferred once a draw or dispatch consumes the writes):\n");
    print!("{}", program.dump());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shaders_reflect_every_supported_scalar_descriptor_type() {
        let bindings = reflected_bindings().unwrap();
        assert!(
            bindings
                .iter()
                .all(|binding| binding.count == 1 && !binding.variable_count)
        );
        let types = bindings
            .into_iter()
            .map(|binding| binding.descriptor_type)
            .collect::<Vec<_>>();

        assert_eq!(
            types,
            [
                vk::DescriptorType::SAMPLER,
                vk::DescriptorType::SAMPLED_IMAGE,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                vk::DescriptorType::STORAGE_IMAGE,
                vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
                vk::DescriptorType::STORAGE_TEXEL_BUFFER,
                vk::DescriptorType::UNIFORM_BUFFER,
                vk::DescriptorType::STORAGE_BUFFER,
                vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                vk::DescriptorType::INPUT_ATTACHMENT,
            ]
        );
    }

    #[test]
    fn every_reflected_type_has_a_compatible_host_payload() {
        let bindings = reflected_bindings().unwrap();
        let program = descriptor_payload_program().unwrap();
        let written = written_descriptors(&program);

        assert_eq!(written.len(), bindings.len());
        for binding in bindings {
            let descriptor = written.get(&(binding.set, binding.binding)).unwrap();
            assert!(
                descriptor.supports_type(binding.descriptor_type),
                "binding {}: {descriptor:?} does not support {:?}",
                binding.binding,
                binding.descriptor_type,
            );
        }
    }
}
