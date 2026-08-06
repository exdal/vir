use std::{collections::HashMap, ffi::CString};

use ash::vk;

const MAGIC: u32 = 0x0723_0203;
const HEADER_WORDS: usize = 5;

mod op {
    pub const ENTRY_POINT: u16 = 15;
    pub const TYPE_VOID: u16 = 19;
    pub const TYPE_BOOL: u16 = 20;
    pub const TYPE_INT: u16 = 21;
    pub const TYPE_FLOAT: u16 = 22;
    pub const TYPE_VECTOR: u16 = 23;
    pub const TYPE_MATRIX: u16 = 24;
    pub const TYPE_IMAGE: u16 = 25;
    pub const TYPE_SAMPLER: u16 = 26;
    pub const TYPE_SAMPLED_IMAGE: u16 = 27;
    pub const TYPE_ARRAY: u16 = 28;
    pub const TYPE_RUNTIME_ARRAY: u16 = 29;
    pub const TYPE_STRUCT: u16 = 30;
    pub const TYPE_POINTER: u16 = 32;
    pub const CONSTANT: u16 = 43;
    pub const VARIABLE: u16 = 59;
    pub const DECORATE: u16 = 71;
    pub const MEMBER_DECORATE: u16 = 72;
    pub const TYPE_ACCELERATION_STRUCTURE: u16 = 5341;
}

mod decoration {
    pub const BUFFER_BLOCK: u32 = 3;
    pub const ARRAY_STRIDE: u32 = 6;
    pub const MATRIX_STRIDE: u32 = 7;
    pub const BUILT_IN: u32 = 11;
    pub const LOCATION: u32 = 30;
    pub const BINDING: u32 = 33;
    pub const DESCRIPTOR_SET: u32 = 34;
    pub const OFFSET: u32 = 35;
}

mod storage_class {
    pub const UNIFORM_CONSTANT: u32 = 0;
    pub const INPUT: u32 = 1;
    pub const UNIFORM: u32 = 2;
    pub const PUSH_CONSTANT: u32 = 9;
    pub const STORAGE_BUFFER: u32 = 12;
}

mod dim {
    pub const BUFFER: u32 = 5;
    pub const SUBPASS_DATA: u32 = 6;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DescriptorBinding {
    pub set: u32,
    pub binding: u32,
    pub descriptor_type: vk::DescriptorType,
    pub count: u32,
    pub variable_count: bool,
}

/// One `Location`-decorated input of a vertex shader, in the order the pipeline packs it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VertexInput {
    pub location: u32,
    pub format: vk::Format,
    pub size: u32,
}

#[derive(Debug, Clone)]
pub struct Reflection {
    pub stage: vk::ShaderStageFlags,
    pub entry_point: CString,
    pub bindings: Vec<DescriptorBinding>,
    /// Where the stage's push constant block starts. Zero when the stage has none.
    pub push_constant_offset: u32,
    /// How many bytes the block spans from `push_constant_offset`. Zero when the stage has none.
    pub push_constant_size: u32,
    /// Empty for every stage but the vertex stage.
    pub vertex_inputs: Vec<VertexInput>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScalarKind {
    Float,
    Sint,
    Uint,
}

#[derive(Debug, Clone)]
enum TypeInfo {
    Scalar { kind: ScalarKind, width: u32 },
    Vector { component: u32, count: u32 },
    Matrix { column: u32, count: u32 },
    Image { dim: u32, sampled: u32 },
    Sampler,
    SampledImage,
    Array { element: u32, length: u32 },
    RuntimeArray { element: u32 },
    Struct { members: Vec<u32> },
    Pointer { pointee: u32 },
    AccelerationStructure,
    Opaque,
}

#[derive(Default)]
struct Parsed {
    entry_points: Vec<(u32, CString)>,
    types: HashMap<u32, TypeInfo>,
    decorations: HashMap<(u32, u32), Vec<u32>>,
    member_decorations: HashMap<(u32, u32, u32), Vec<u32>>,
    constants: HashMap<u32, u32>,
    variables: Vec<(u32, u32, u32)>,
}

fn decode_literal_string(words: &[u32]) -> Option<(CString, usize)> {
    let mut bytes = Vec::new();
    for (index, word) in words.iter().enumerate() {
        for shift in [0, 8, 16, 24] {
            let byte = ((word >> shift) & 0xff) as u8;
            if byte == 0 {
                return CString::new(bytes).ok().map(|s| (s, index + 1));
            }
            bytes.push(byte);
        }
    }
    None
}

fn parse(spirv: &[u32]) -> Option<Parsed> {
    if spirv.len() < HEADER_WORDS || spirv[0] != MAGIC {
        return None;
    }

    let mut parsed = Parsed::default();
    let mut cursor = HEADER_WORDS;

    while cursor < spirv.len() {
        let word_count = (spirv[cursor] >> 16) as usize;
        let opcode = (spirv[cursor] & 0xffff) as u16;
        if word_count == 0 || cursor + word_count > spirv.len() {
            return None;
        }
        let operands = &spirv[cursor + 1..cursor + word_count];

        match opcode {
            op::ENTRY_POINT if operands.len() >= 3 => {
                let (name, _) = decode_literal_string(&operands[2..])?;
                parsed.entry_points.push((operands[0], name));
            },
            op::TYPE_VOID | op::TYPE_BOOL if !operands.is_empty() => {
                parsed.types.insert(operands[0], TypeInfo::Opaque);
            },
            op::TYPE_FLOAT if operands.len() >= 2 => {
                parsed.types.insert(
                    operands[0],
                    TypeInfo::Scalar {
                        kind: ScalarKind::Float,
                        width: operands[1],
                    },
                );
            },
            op::TYPE_INT if operands.len() >= 3 => {
                let kind = if operands[2] == 0 {
                    ScalarKind::Uint
                } else {
                    ScalarKind::Sint
                };
                parsed.types.insert(
                    operands[0],
                    TypeInfo::Scalar {
                        kind,
                        width: operands[1],
                    },
                );
            },
            op::TYPE_VECTOR if operands.len() >= 3 => {
                parsed.types.insert(
                    operands[0],
                    TypeInfo::Vector {
                        component: operands[1],
                        count: operands[2],
                    },
                );
            },
            op::TYPE_MATRIX if operands.len() >= 3 => {
                parsed.types.insert(
                    operands[0],
                    TypeInfo::Matrix {
                        column: operands[1],
                        count: operands[2],
                    },
                );
            },
            op::TYPE_IMAGE if operands.len() >= 7 => {
                parsed.types.insert(
                    operands[0],
                    TypeInfo::Image {
                        dim: operands[2],
                        sampled: operands[6],
                    },
                );
            },
            op::TYPE_SAMPLER if !operands.is_empty() => {
                parsed.types.insert(operands[0], TypeInfo::Sampler);
            },
            op::TYPE_SAMPLED_IMAGE if !operands.is_empty() => {
                parsed.types.insert(operands[0], TypeInfo::SampledImage);
            },
            op::TYPE_ARRAY if operands.len() >= 3 => {
                parsed.types.insert(
                    operands[0],
                    TypeInfo::Array {
                        element: operands[1],
                        length: operands[2],
                    },
                );
            },
            op::TYPE_RUNTIME_ARRAY if operands.len() >= 2 => {
                parsed
                    .types
                    .insert(operands[0], TypeInfo::RuntimeArray { element: operands[1] });
            },
            op::TYPE_STRUCT if !operands.is_empty() => {
                parsed.types.insert(
                    operands[0],
                    TypeInfo::Struct {
                        members: operands[1..].to_vec(),
                    },
                );
            },
            op::TYPE_POINTER if operands.len() >= 3 => {
                parsed
                    .types
                    .insert(operands[0], TypeInfo::Pointer { pointee: operands[2] });
            },
            op::TYPE_ACCELERATION_STRUCTURE if !operands.is_empty() => {
                parsed.types.insert(operands[0], TypeInfo::AccelerationStructure);
            },
            op::CONSTANT if operands.len() >= 3 => {
                parsed.constants.insert(operands[1], operands[2]);
            },
            op::VARIABLE if operands.len() >= 3 => {
                parsed.variables.push((operands[1], operands[0], operands[2]));
            },
            op::DECORATE if operands.len() >= 2 => {
                parsed
                    .decorations
                    .insert((operands[0], operands[1]), operands[2..].to_vec());
            },
            op::MEMBER_DECORATE if operands.len() >= 3 => {
                parsed
                    .member_decorations
                    .insert((operands[0], operands[1], operands[2]), operands[3..].to_vec());
            },
            _ => {},
        }

        cursor += word_count;
    }

    Some(parsed)
}

impl Parsed {
    fn decoration(&self, target: u32, decoration: u32) -> Option<u32> {
        self.decorations.get(&(target, decoration))?.first().copied()
    }

    fn has_decoration(&self, target: u32, decoration: u32) -> bool {
        self.decorations.contains_key(&(target, decoration))
    }

    fn member_decoration(&self, target: u32, member: u32, decoration: u32) -> Option<u32> {
        self.member_decorations
            .get(&(target, member, decoration))?
            .first()
            .copied()
    }

    fn peel_arrays(&self, mut ty: u32) -> (u32, u32, bool) {
        let mut count = 1u32;
        let mut variable = false;

        for _ in 0..64 {
            match self.types.get(&ty) {
                Some(TypeInfo::Array { element, length }) => {
                    count = count.saturating_mul(self.constants.get(length).copied().unwrap_or(1));
                    ty = *element;
                },
                Some(TypeInfo::RuntimeArray { element }) => {
                    variable = true;
                    ty = *element;
                },
                _ => break,
            }
        }

        (ty, count, variable)
    }

    fn type_size(&self, ty: u32, depth: u32) -> u32 {
        if depth > 64 {
            return 0;
        }

        match self.types.get(&ty) {
            Some(TypeInfo::Scalar { width, .. }) => width / 8,
            Some(TypeInfo::Vector { component, count }) => self.type_size(*component, depth + 1) * count,
            Some(TypeInfo::Matrix { column, count }) => {
                self.decoration(ty, decoration::MATRIX_STRIDE)
                    .unwrap_or_else(|| self.type_size(*column, depth + 1))
                    * count
            },
            Some(TypeInfo::Array { element, length }) => {
                let length = self.constants.get(length).copied().unwrap_or(0);
                let stride = self
                    .decoration(ty, decoration::ARRAY_STRIDE)
                    .unwrap_or_else(|| self.type_size(*element, depth + 1));
                stride * length
            },
            Some(TypeInfo::Pointer { .. }) => 8,
            Some(TypeInfo::Struct { members }) => members
                .iter()
                .enumerate()
                .map(|(index, member)| {
                    let offset = self
                        .member_decoration(ty, index as u32, decoration::OFFSET)
                        .unwrap_or(0);
                    offset + self.type_size(*member, depth + 1)
                })
                .max()
                .unwrap_or(0),
            _ => 0,
        }
    }

    /// The first byte a push constant block occupies.
    ///
    /// A block whose members all sit past some offset only needs a range that starts there, which
    /// is what the shader was compiled against.
    fn block_start(&self, ty: u32) -> u32 {
        match self.types.get(&ty) {
            Some(TypeInfo::Struct { members }) => (0..members.len() as u32)
                .map(|member| self.member_decoration(ty, member, decoration::OFFSET).unwrap_or(0))
                .min()
                .unwrap_or(0),
            _ => 0,
        }
    }

    /// The attribute format a vertex input of this type is read through.
    ///
    /// Only 32-bit components are handled: anything wider, and matrices, span more than one
    /// location, which the single-binding packing below has no way to express.
    fn vertex_format(&self, ty: u32) -> Option<vk::Format> {
        let (kind, width, count) = match self.types.get(&ty)? {
            TypeInfo::Scalar { kind, width } => (*kind, *width, 1),
            TypeInfo::Vector { component, count } => match self.types.get(component)? {
                TypeInfo::Scalar { kind, width } => (*kind, *width, *count),
                _ => return None,
            },
            _ => return None,
        };

        if width != 32 {
            return None;
        }

        Some(match (kind, count) {
            (ScalarKind::Float, 1) => vk::Format::R32_SFLOAT,
            (ScalarKind::Float, 2) => vk::Format::R32G32_SFLOAT,
            (ScalarKind::Float, 3) => vk::Format::R32G32B32_SFLOAT,
            (ScalarKind::Float, 4) => vk::Format::R32G32B32A32_SFLOAT,
            (ScalarKind::Sint, 1) => vk::Format::R32_SINT,
            (ScalarKind::Sint, 2) => vk::Format::R32G32_SINT,
            (ScalarKind::Sint, 3) => vk::Format::R32G32B32_SINT,
            (ScalarKind::Sint, 4) => vk::Format::R32G32B32A32_SINT,
            (ScalarKind::Uint, 1) => vk::Format::R32_UINT,
            (ScalarKind::Uint, 2) => vk::Format::R32G32_UINT,
            (ScalarKind::Uint, 3) => vk::Format::R32G32B32_UINT,
            (ScalarKind::Uint, 4) => vk::Format::R32G32B32A32_UINT,
            _ => return None,
        })
    }

    fn descriptor_type(&self, storage_class: u32, pointee: u32) -> Option<vk::DescriptorType> {
        match storage_class {
            storage_class::UNIFORM_CONSTANT => match self.types.get(&pointee)? {
                TypeInfo::Image { dim, sampled } => Some(match (*dim, *sampled) {
                    (dim::SUBPASS_DATA, _) => vk::DescriptorType::INPUT_ATTACHMENT,
                    (dim::BUFFER, 1) => vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
                    (dim::BUFFER, _) => vk::DescriptorType::STORAGE_TEXEL_BUFFER,
                    (_, 1) => vk::DescriptorType::SAMPLED_IMAGE,
                    _ => vk::DescriptorType::STORAGE_IMAGE,
                }),
                TypeInfo::Sampler => Some(vk::DescriptorType::SAMPLER),
                TypeInfo::SampledImage => Some(vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
                TypeInfo::AccelerationStructure => Some(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR),
                _ => None,
            },
            storage_class::UNIFORM => Some(if self.has_decoration(pointee, decoration::BUFFER_BLOCK) {
                vk::DescriptorType::STORAGE_BUFFER
            } else {
                vk::DescriptorType::UNIFORM_BUFFER
            }),
            storage_class::STORAGE_BUFFER => Some(vk::DescriptorType::STORAGE_BUFFER),
            _ => None,
        }
    }
}

fn stage_from_execution_model(model: u32) -> Option<vk::ShaderStageFlags> {
    Some(match model {
        0 => vk::ShaderStageFlags::VERTEX,
        1 => vk::ShaderStageFlags::TESSELLATION_CONTROL,
        2 => vk::ShaderStageFlags::TESSELLATION_EVALUATION,
        3 => vk::ShaderStageFlags::GEOMETRY,
        4 => vk::ShaderStageFlags::FRAGMENT,
        5 => vk::ShaderStageFlags::COMPUTE,
        _ => return None,
    })
}

pub fn reflect(spirv: &[u32]) -> Result<Reflection, vk::Result> {
    let Some(parsed) = parse(spirv) else {
        tracing::error!("shader is not a well-formed SPIR-V module");
        return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
    };

    let [(execution_model, entry_point)] = parsed.entry_points.as_slice() else {
        tracing::error!(
            count = parsed.entry_points.len(),
            "shader must declare exactly one entry point"
        );
        return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
    };

    let Some(stage) = stage_from_execution_model(*execution_model) else {
        tracing::error!(execution_model, "unsupported SPIR-V execution model");
        return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
    };

    let mut bindings = Vec::new();
    let mut vertex_inputs = Vec::new();
    // the stage's push constants as a half-open byte range; `None` until one shows up
    let mut push_constants: Option<(u32, u32)> = None;

    for (variable, pointer_type, storage_class) in &parsed.variables {
        let Some(TypeInfo::Pointer { pointee, .. }) = parsed.types.get(pointer_type) else {
            continue;
        };

        if *storage_class == storage_class::PUSH_CONSTANT {
            let start = parsed.block_start(*pointee);
            let end = parsed.type_size(*pointee, 0);
            if end > start {
                push_constants = Some(match push_constants {
                    Some((first, last)) => (first.min(start), last.max(end)),
                    None => (start, end),
                });
            }
            continue;
        }

        // fragment stages have `Input` variables too, but those are interpolants, not attributes
        if *storage_class == storage_class::INPUT {
            if stage != vk::ShaderStageFlags::VERTEX {
                continue;
            }
            // builtins like SV_VertexID arrive on `Input` but are fed by the driver
            if parsed.has_decoration(*variable, decoration::BUILT_IN) {
                continue;
            }
            let Some(location) = parsed.decoration(*variable, decoration::LOCATION) else {
                continue;
            };
            let Some(format) = parsed.vertex_format(*pointee) else {
                tracing::warn!(location, "unsupported vertex input type; the attribute is dropped");
                continue;
            };

            vertex_inputs.push(VertexInput {
                location,
                format,
                size: parsed.type_size(*pointee, 0),
            });
            continue;
        }

        let (Some(set), Some(binding)) = (
            parsed.decoration(*variable, decoration::DESCRIPTOR_SET),
            parsed.decoration(*variable, decoration::BINDING),
        ) else {
            continue;
        };

        let (element, count, variable_count) = parsed.peel_arrays(*pointee);
        let Some(descriptor_type) = parsed.descriptor_type(*storage_class, element) else {
            continue;
        };

        bindings.push(DescriptorBinding {
            set,
            binding,
            descriptor_type,
            count,
            variable_count,
        });
    }

    bindings.sort_unstable_by_key(|b| (b.set, b.binding));
    vertex_inputs.sort_unstable_by_key(|input| input.location);

    let (push_constant_offset, push_constant_size) = match push_constants {
        Some((start, end)) => (start, end - start),
        None => (0, 0),
    };

    Ok(Reflection {
        stage,
        entry_point: entry_point.clone(),
        bindings,
        push_constant_offset,
        push_constant_size,
        vertex_inputs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inst(opcode: u16, operands: &[u32]) -> Vec<u32> {
        let mut words = vec![((operands.len() as u32 + 1) << 16) | opcode as u32];
        words.extend_from_slice(operands);
        words
    }

    fn literal(text: &str) -> Vec<u32> {
        let mut bytes = text.as_bytes().to_vec();
        bytes.push(0);
        bytes.resize(bytes.len().div_ceil(4) * 4, 0);
        bytes
            .chunks(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn module() -> Vec<u32> {
        let mut words = vec![MAGIC, 0x0001_0300, 0, 100, 0];

        let mut entry = vec![4, 99];
        entry.extend(literal("main"));
        words.extend(inst(op::ENTRY_POINT, &entry));

        words.extend(inst(op::DECORATE, &[5, decoration::DESCRIPTOR_SET, 0]));
        words.extend(inst(op::DECORATE, &[5, decoration::BINDING, 0]));
        words.extend(inst(op::DECORATE, &[9, decoration::DESCRIPTOR_SET, 1]));
        words.extend(inst(op::DECORATE, &[9, decoration::BINDING, 0]));
        words.extend(inst(op::DECORATE, &[15, decoration::DESCRIPTOR_SET, 1]));
        words.extend(inst(op::DECORATE, &[15, decoration::BINDING, 1]));
        words.extend(inst(op::DECORATE, &[21, decoration::DESCRIPTOR_SET, 0]));
        words.extend(inst(op::DECORATE, &[21, decoration::BINDING, 2]));
        words.extend(inst(op::MEMBER_DECORATE, &[16, 0, decoration::OFFSET, 0]));
        words.extend(inst(op::MEMBER_DECORATE, &[16, 1, decoration::OFFSET, 16]));

        words.extend(inst(op::TYPE_FLOAT, &[1, 32]));
        words.extend(inst(op::TYPE_VECTOR, &[2, 1, 4]));

        words.extend(inst(op::TYPE_STRUCT, &[3, 2]));
        words.extend(inst(op::TYPE_POINTER, &[4, storage_class::UNIFORM, 3]));
        words.extend(inst(op::VARIABLE, &[4, 5, storage_class::UNIFORM]));

        words.extend(inst(op::TYPE_IMAGE, &[6, 1, 1, 0, 0, 0, 1, 0]));
        words.extend(inst(op::TYPE_RUNTIME_ARRAY, &[7, 6]));
        words.extend(inst(op::TYPE_POINTER, &[8, storage_class::UNIFORM_CONSTANT, 7]));
        words.extend(inst(op::VARIABLE, &[8, 9, storage_class::UNIFORM_CONSTANT]));

        words.extend(inst(op::TYPE_SAMPLER, &[10]));
        words.extend(inst(op::TYPE_INT, &[11, 32, 0]));
        words.extend(inst(op::CONSTANT, &[11, 12, 4]));
        words.extend(inst(op::TYPE_ARRAY, &[13, 10, 12]));
        words.extend(inst(op::TYPE_POINTER, &[14, storage_class::UNIFORM_CONSTANT, 13]));
        words.extend(inst(op::VARIABLE, &[14, 15, storage_class::UNIFORM_CONSTANT]));

        words.extend(inst(op::TYPE_STRUCT, &[16, 2, 1]));
        words.extend(inst(op::TYPE_POINTER, &[17, storage_class::PUSH_CONSTANT, 16]));
        words.extend(inst(op::VARIABLE, &[17, 18, storage_class::PUSH_CONSTANT]));

        words.extend(inst(op::TYPE_STRUCT, &[19, 1]));
        words.extend(inst(op::TYPE_POINTER, &[20, storage_class::STORAGE_BUFFER, 19]));
        words.extend(inst(op::VARIABLE, &[20, 21, storage_class::STORAGE_BUFFER]));

        words.extend(inst(op::TYPE_POINTER, &[22, 3, 2]));
        words.extend(inst(op::VARIABLE, &[22, 23, 3]));

        words
    }

    #[test]
    fn reads_stage_and_entry_point() {
        let reflection = reflect(&module()).expect("module should reflect");
        assert_eq!(reflection.stage, vk::ShaderStageFlags::FRAGMENT);
        assert_eq!(reflection.entry_point.to_str().unwrap(), "main");
    }

    #[test]
    fn reads_bindings_in_set_and_binding_order() {
        let reflection = reflect(&module()).expect("module should reflect");
        let actual: Vec<_> = reflection
            .bindings
            .iter()
            .map(|b| (b.set, b.binding, b.descriptor_type, b.count, b.variable_count))
            .collect();

        assert_eq!(
            actual,
            vec![
                (0, 0, vk::DescriptorType::UNIFORM_BUFFER, 1, false),
                (0, 2, vk::DescriptorType::STORAGE_BUFFER, 1, false),
                (1, 0, vk::DescriptorType::SAMPLED_IMAGE, 1, true),
                (1, 1, vk::DescriptorType::SAMPLER, 4, false),
            ]
        );
    }

    #[test]
    fn sizes_push_constants_from_member_offsets() {
        let reflection = reflect(&module()).expect("module should reflect");
        assert_eq!(reflection.push_constant_offset, 0);
        assert_eq!(reflection.push_constant_size, 20);
    }

    #[test]
    fn a_push_constant_block_that_starts_late_reports_the_offset_it_starts_at() {
        let mut words = module();
        // move the block's first member from 0 to 32, past the second one
        let member_offset = words
            .windows(5)
            .position(|w| w[0] == (5 << 16) | op::MEMBER_DECORATE as u32 && w[1..4] == [16, 0, decoration::OFFSET])
            .expect("member decoration should be present");
        words[member_offset + 4] = 32;

        let reflection = reflect(&words).expect("module should reflect");
        // members at 16 and 32, the widest ending at 48
        assert_eq!(reflection.push_constant_offset, 16);
        assert_eq!(reflection.push_constant_size, 32);
    }

    #[test]
    fn a_shader_with_no_push_constants_reports_an_empty_range() {
        let reflection = reflect(&vertex_module()).expect("module should reflect");
        assert_eq!(reflection.push_constant_offset, 0);
        assert_eq!(reflection.push_constant_size, 0);
    }

    #[test]
    fn ignores_stage_inputs_outside_the_vertex_stage() {
        let reflection = reflect(&module()).expect("module should reflect");
        assert_eq!(reflection.stage, vk::ShaderStageFlags::FRAGMENT);
        assert!(reflection.vertex_inputs.is_empty());
    }

    /// A vertex module with two `Location`-decorated inputs and one builtin.
    fn vertex_module() -> Vec<u32> {
        let mut words = vec![MAGIC, 0x0001_0300, 0, 100, 0];

        let mut entry = vec![0, 99];
        entry.extend(literal("vs_main"));
        words.extend(inst(op::ENTRY_POINT, &entry));

        // declared out of location order, to prove the result is sorted
        words.extend(inst(op::DECORATE, &[9, decoration::LOCATION, 1]));
        words.extend(inst(op::DECORATE, &[5, decoration::LOCATION, 0]));
        // SV_VertexID lands here: an `Input` variable that the driver feeds
        words.extend(inst(op::DECORATE, &[13, decoration::BUILT_IN, 42]));

        words.extend(inst(op::TYPE_FLOAT, &[1, 32]));
        words.extend(inst(op::TYPE_VECTOR, &[2, 1, 2]));
        words.extend(inst(op::TYPE_VECTOR, &[3, 1, 3]));
        words.extend(inst(op::TYPE_INT, &[11, 32, 0]));

        words.extend(inst(op::TYPE_POINTER, &[4, storage_class::INPUT, 2]));
        words.extend(inst(op::VARIABLE, &[4, 5, storage_class::INPUT]));

        words.extend(inst(op::TYPE_POINTER, &[8, storage_class::INPUT, 3]));
        words.extend(inst(op::VARIABLE, &[8, 9, storage_class::INPUT]));

        words.extend(inst(op::TYPE_POINTER, &[12, storage_class::INPUT, 11]));
        words.extend(inst(op::VARIABLE, &[12, 13, storage_class::INPUT]));

        // an unsupported input: a 4x4 matrix spans four locations
        words.extend(inst(op::DECORATE, &[16, decoration::LOCATION, 4]));
        words.extend(inst(op::TYPE_VECTOR, &[14, 1, 4]));
        words.extend(inst(op::TYPE_MATRIX, &[17, 14, 4]));
        words.extend(inst(op::TYPE_POINTER, &[15, storage_class::INPUT, 17]));
        words.extend(inst(op::VARIABLE, &[15, 16, storage_class::INPUT]));

        words
    }

    #[test]
    fn reads_vertex_inputs_in_location_order() {
        let reflection = reflect(&vertex_module()).expect("module should reflect");
        assert_eq!(reflection.stage, vk::ShaderStageFlags::VERTEX);
        assert_eq!(
            reflection.vertex_inputs,
            vec![
                VertexInput {
                    location: 0,
                    format: vk::Format::R32G32_SFLOAT,
                    size: 8,
                },
                VertexInput {
                    location: 1,
                    format: vk::Format::R32G32B32_SFLOAT,
                    size: 12,
                },
            ]
        );
    }

    #[test]
    fn rejects_a_module_that_is_not_spirv() {
        assert!(reflect(&[0, 1, 2, 3, 4]).is_err());
    }

    #[test]
    fn rejects_a_module_with_no_entry_point() {
        let mut words = module();
        let entry_len = 3 + literal("main").len();
        words.splice(5..5 + entry_len, std::iter::empty());
        assert!(reflect(&words).is_err());
    }
}
