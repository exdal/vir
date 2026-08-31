use std::{
    collections::{HashMap, HashSet},
    ffi::CString,
};

use ash::vk;

use crate::Access;

const MAGIC: u32 = 0x0723_0203;
const HEADER_WORDS: usize = 5;

mod op {
    pub const NOP: u16 = 0;
    pub const LINE: u16 = 8;
    pub const EXT_INST: u16 = 12;
    pub const ENTRY_POINT: u16 = 15;
    pub const EXECUTION_MODE: u16 = 16;
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
    #[cfg(test)]
    pub const TYPE_FUNCTION: u16 = 33;
    pub const CONSTANT: u16 = 43;
    pub const FUNCTION: u16 = 54;
    pub const FUNCTION_PARAMETER: u16 = 55;
    pub const FUNCTION_END: u16 = 56;
    pub const FUNCTION_CALL: u16 = 57;
    pub const VARIABLE: u16 = 59;
    pub const IMAGE_TEXEL_POINTER: u16 = 60;
    pub const LOAD: u16 = 61;
    pub const STORE: u16 = 62;
    pub const COPY_MEMORY: u16 = 63;
    pub const COPY_MEMORY_SIZED: u16 = 64;
    pub const ACCESS_CHAIN: u16 = 65;
    pub const IN_BOUNDS_ACCESS_CHAIN: u16 = 66;
    pub const PTR_ACCESS_CHAIN: u16 = 67;
    pub const ARRAY_LENGTH: u16 = 68;
    pub const IN_BOUNDS_PTR_ACCESS_CHAIN: u16 = 70;
    pub const DECORATE: u16 = 71;
    pub const MEMBER_DECORATE: u16 = 72;
    pub const VECTOR_EXTRACT_DYNAMIC: u16 = 77;
    pub const COMPOSITE_INSERT: u16 = 82;
    pub const COPY_OBJECT: u16 = 83;
    pub const TRANSPOSE: u16 = 84;
    pub const SAMPLED_IMAGE: u16 = 86;
    pub const IMAGE_SAMPLE_FIRST: u16 = 87;
    pub const IMAGE_READ: u16 = 98;
    pub const IMAGE_WRITE: u16 = 99;
    pub const IMAGE: u16 = 100;
    pub const IMAGE_QUERY_FORMAT: u16 = 101;
    pub const IMAGE_QUERY_SAMPLES: u16 = 107;
    pub const BITCAST: u16 = 124;
    pub const SELECT: u16 = 169;
    pub const ATOMIC_LOAD: u16 = 227;
    pub const ATOMIC_STORE: u16 = 228;
    pub const ATOMIC_RMW_FIRST: u16 = 229;
    pub const ATOMIC_RMW_LAST: u16 = 242;
    pub const PHI: u16 = 245;
    pub const LOOP_MERGE: u16 = 246;
    pub const SELECTION_MERGE: u16 = 247;
    pub const LABEL: u16 = 248;
    pub const BRANCH: u16 = 249;
    pub const BRANCH_CONDITIONAL: u16 = 250;
    pub const SWITCH: u16 = 251;
    pub const KILL: u16 = 252;
    pub const RETURN: u16 = 253;
    pub const RETURN_VALUE: u16 = 254;
    pub const UNREACHABLE: u16 = 255;
    pub const IMAGE_SPARSE_SAMPLE_FIRST: u16 = 305;
    pub const IMAGE_SPARSE_DREF_GATHER: u16 = 315;
    pub const ATOMIC_FLAG_TEST_AND_SET: u16 = 318;
    pub const ATOMIC_FLAG_CLEAR: u16 = 319;
    pub const NO_LINE: u16 = 317;
    pub const IMAGE_SPARSE_READ: u16 = 320;
    pub const EXECUTION_MODE_ID: u16 = 331;
    pub const COPY_LOGICAL: u16 = 400;
    pub const TRACE_RAY: u16 = 4445;
    pub const RAY_QUERY_INITIALIZE: u16 = 4473;
    pub const IMAGE_SAMPLE_FOOTPRINT: u16 = 5283;
    pub const TYPE_ACCELERATION_STRUCTURE: u16 = 5341;
    pub const ATOMIC_FMIN: u16 = 5614;
    pub const ATOMIC_FMAX: u16 = 5615;
    pub const ATOMIC_FADD: u16 = 6035;
}

mod execution_mode {
    pub const LOCAL_SIZE: u32 = 17;
    pub const LOCAL_SIZE_ID: u32 = 38;
}

mod decoration {
    pub const BUFFER_BLOCK: u32 = 3;
    pub const ARRAY_STRIDE: u32 = 6;
    pub const MATRIX_STRIDE: u32 = 7;
    pub const BUILT_IN: u32 = 11;
    pub const NON_WRITABLE: u32 = 24;
    pub const NON_READABLE: u32 = 25;
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
    /// Every pipeline stage whose module declares this binding.
    pub stages: vk::ShaderStageFlags,
    /// The descriptor memory operations in this entry point's static call tree.
    ///
    /// Runtime branches are all included; a binding with no reachable operation is [`Access::None`].
    pub access: Access,
}

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
    /// The workgroup the stage declares. `[1, 1, 1]` for every stage but the compute stage.
    pub local_size: [u32; 3],
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

struct EntryPoint {
    execution_model: u32,
    function: u32,
    name: CString,
}

struct Instruction {
    opcode: u16,
    operands: Vec<u32>,
    function: u32,
}

struct PendingDescriptor {
    variable: u32,
    pointer_type: u32,
    binding: DescriptorBinding,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Origin {
    variable: u32,
    non_readable: bool,
    non_writable: bool,
}

#[derive(Debug, Clone, Copy)]
enum DescriptorOperation {
    MemoryRead,
    MemoryWrite,
    ImageRead,
    ImageWrite,
    AccelerationStructureRead,
}

#[derive(Default)]
struct Parsed {
    entry_points: Vec<EntryPoint>,
    types: HashMap<u32, TypeInfo>,
    decorations: HashMap<(u32, u32), Vec<u32>>,
    member_decorations: HashMap<(u32, u32, u32), Vec<u32>>,
    constants: HashMap<u32, u32>,
    variables: Vec<(u32, u32, u32)>,
    value_types: HashMap<u32, u32>,
    instructions: Vec<Instruction>,
    local_size: Option<[u32; 3]>,
    /// A `LocalSizeId` names constants rather than literals, which are only known once the whole
    /// module has been walked.
    local_size_id: Option<[u32; 3]>,
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
    let mut current_function = None;

    while cursor < spirv.len() {
        let word_count = (spirv[cursor] >> 16) as usize;
        let opcode = (spirv[cursor] & 0xffff) as u16;
        if word_count == 0 || cursor + word_count > spirv.len() {
            return None;
        }
        let operands = &spirv[cursor + 1..cursor + word_count];

        if opcode == op::FUNCTION {
            if operands.len() < 2 || current_function.is_some() {
                return None;
            }
            current_function = Some(operands[1]);
        }
        if let Some(function) = current_function {
            parsed.instructions.push(Instruction {
                opcode,
                operands: operands.to_vec(),
                function,
            });
        }

        match opcode {
            op::ENTRY_POINT if operands.len() >= 3 => {
                let (name, _) = decode_literal_string(&operands[2..])?;
                parsed.entry_points.push(EntryPoint {
                    execution_model: operands[0],
                    function: operands[1],
                    name,
                });
            },
            op::EXECUTION_MODE if operands.len() >= 5 && operands[1] == execution_mode::LOCAL_SIZE => {
                parsed.local_size = Some([operands[2], operands[3], operands[4]]);
            },
            op::EXECUTION_MODE_ID if operands.len() >= 5 && operands[1] == execution_mode::LOCAL_SIZE_ID => {
                parsed.local_size_id = Some([operands[2], operands[3], operands[4]]);
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
                parsed.value_types.insert(operands[1], operands[0]);
            },
            op::FUNCTION_PARAMETER
            | op::FUNCTION_CALL
            | op::IMAGE_TEXEL_POINTER
            | op::LOAD
            | op::ACCESS_CHAIN
            | op::IN_BOUNDS_ACCESS_CHAIN
            | op::PTR_ACCESS_CHAIN
            | op::IN_BOUNDS_PTR_ACCESS_CHAIN
            | op::COPY_OBJECT
            | op::COPY_LOGICAL
            | op::SAMPLED_IMAGE
            | op::IMAGE
            | op::BITCAST
            | op::SELECT
            | op::PHI
                if operands.len() >= 2 =>
            {
                parsed.value_types.insert(operands[1], operands[0]);
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

        if opcode == op::FUNCTION_END {
            current_function = None;
        }

        cursor += word_count;
    }

    if current_function.is_some() {
        return None;
    }

    Some(parsed)
}

impl Parsed {
    /// The workgroup the module declares, whether it spelled it as literals or as constants.
    fn declared_local_size(&self) -> Option<[u32; 3]> {
        if let Some(local_size) = self.local_size {
            return Some(local_size);
        }

        let ids = self.local_size_id?;
        Some([
            *self.constants.get(&ids[0])?,
            *self.constants.get(&ids[1])?,
            *self.constants.get(&ids[2])?,
        ])
    }

    fn decoration(&self, target: u32, decoration: u32) -> Option<u32> {
        self.decorations.get(&(target, decoration))?.first().copied()
    }

    fn has_decoration(&self, target: u32, decoration: u32) -> bool {
        self.decorations.contains_key(&(target, decoration))
    }

    fn has_member_decoration(&self, target: u32, member: u32, decoration: u32) -> bool {
        self.member_decorations.contains_key(&(target, member, decoration))
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

    fn type_constraints(&self, ty: u32, depth: u32) -> (bool, bool) {
        if depth > 64 {
            return (false, false);
        }

        let direct = (
            self.has_decoration(ty, decoration::NON_READABLE),
            self.has_decoration(ty, decoration::NON_WRITABLE),
        );
        let nested = match self.types.get(&ty) {
            Some(TypeInfo::Pointer { pointee }) => self.type_constraints(*pointee, depth + 1),
            Some(TypeInfo::Array { element, .. }) | Some(TypeInfo::RuntimeArray { element }) => {
                self.type_constraints(*element, depth + 1)
            },
            _ => (false, false),
        };
        (direct.0 || nested.0, direct.1 || nested.1)
    }

    fn type_carries_provenance(&self, ty: u32, depth: u32) -> bool {
        if depth > 64 {
            return false;
        }

        match self.types.get(&ty) {
            Some(
                TypeInfo::Pointer { .. }
                | TypeInfo::Image { .. }
                | TypeInfo::Sampler
                | TypeInfo::SampledImage
                | TypeInfo::AccelerationStructure,
            ) => true,
            Some(TypeInfo::Array { element, .. } | TypeInfo::RuntimeArray { element }) => {
                self.type_carries_provenance(*element, depth + 1)
            },
            _ => false,
        }
    }

    fn value_carries_provenance(&self, value: u32) -> bool {
        self.value_types
            .get(&value)
            .is_some_and(|ty| self.type_carries_provenance(*ty, 0))
    }

    fn value_is_opaque_handle(&self, value: u32) -> bool {
        self.value_types.get(&value).is_some_and(|ty| {
            matches!(
                self.types.get(ty),
                Some(
                    TypeInfo::Image { .. }
                        | TypeInfo::Sampler
                        | TypeInfo::SampledImage
                        | TypeInfo::AccelerationStructure
                )
            )
        })
    }

    fn reachable_functions(&self, entry: u32) -> HashSet<u32> {
        let mut reachable = HashSet::from([entry]);
        loop {
            let before = reachable.len();
            for instruction in &self.instructions {
                if instruction.opcode == op::FUNCTION_CALL
                    && reachable.contains(&instruction.function)
                    && instruction.operands.len() >= 3
                {
                    reachable.insert(instruction.operands[2]);
                }
            }
            if reachable.len() == before {
                return reachable;
            }
        }
    }

    fn access_chain_constraints(&self, base: u32, indices: &[u32]) -> (bool, bool) {
        let Some(mut ty) = self.value_types.get(&base).copied() else {
            return (false, false);
        };
        let mut constraints = (false, false);

        for index in indices {
            for _ in 0..64 {
                match self.types.get(&ty) {
                    Some(TypeInfo::Pointer { pointee }) => ty = *pointee,
                    _ => break,
                }
            }

            match self.types.get(&ty) {
                Some(TypeInfo::Struct { members }) => {
                    let Some(member) = self.constants.get(index).copied() else {
                        continue;
                    };
                    let Some(member_type) = members.get(member as usize).copied() else {
                        continue;
                    };
                    constraints.0 |= self.has_member_decoration(ty, member, decoration::NON_READABLE);
                    constraints.1 |= self.has_member_decoration(ty, member, decoration::NON_WRITABLE);
                    let nested = self.type_constraints(member_type, 0);
                    constraints.0 |= nested.0;
                    constraints.1 |= nested.1;
                    ty = member_type;
                },
                Some(TypeInfo::Array { element, .. }) | Some(TypeInfo::RuntimeArray { element }) => {
                    ty = *element;
                },
                Some(TypeInfo::Vector { component, .. }) => ty = *component,
                Some(TypeInfo::Matrix { column, .. }) => ty = *column,
                _ => {},
            }
        }

        constraints
    }
}

type Provenance = HashMap<u32, HashSet<Origin>>;

fn origins_from(provenance: &Provenance, sources: impl IntoIterator<Item = u32>) -> HashSet<Origin> {
    sources
        .into_iter()
        .filter_map(|source| provenance.get(&source))
        .flat_map(|origins| origins.iter().copied())
        .collect()
}

fn extend_origins(provenance: &mut Provenance, result: u32, additions: HashSet<Origin>) -> bool {
    if additions.is_empty() {
        return false;
    }

    let result_origins = provenance.entry(result).or_default();
    let before = result_origins.len();
    result_origins.extend(additions);
    result_origins.len() != before
}

impl Parsed {
    fn descriptor_accesses(
        &self, entry_function: u32, stage: vk::ShaderStageFlags, descriptors: &[PendingDescriptor],
    ) -> Result<HashMap<u32, Access>, vk::Result> {
        let descriptors_by_variable: HashMap<_, _> = descriptors
            .iter()
            .map(|descriptor| (descriptor.variable, descriptor))
            .collect();
        let reachable = self.reachable_functions(entry_function);
        let mut provenance = Provenance::new();

        for descriptor in descriptors {
            let type_constraints = self.type_constraints(descriptor.pointer_type, 0);
            provenance.insert(
                descriptor.variable,
                HashSet::from([Origin {
                    variable: descriptor.variable,
                    non_readable: self.has_decoration(descriptor.variable, decoration::NON_READABLE)
                        || type_constraints.0,
                    non_writable: self.has_decoration(descriptor.variable, decoration::NON_WRITABLE)
                        || type_constraints.1,
                }]),
            );
        }

        let mut parameters: HashMap<u32, Vec<u32>> = HashMap::new();
        let mut return_values: HashMap<u32, Vec<u32>> = HashMap::new();
        for instruction in &self.instructions {
            match instruction.opcode {
                op::FUNCTION_PARAMETER if instruction.operands.len() >= 2 => parameters
                    .entry(instruction.function)
                    .or_default()
                    .push(instruction.operands[1]),
                op::RETURN_VALUE if !instruction.operands.is_empty() => return_values
                    .entry(instruction.function)
                    .or_default()
                    .push(instruction.operands[0]),
                _ => {},
            }
        }

        loop {
            let mut changed = false;
            for instruction in self
                .instructions
                .iter()
                .filter(|instruction| reachable.contains(&instruction.function))
            {
                let operands = instruction.operands.as_slice();
                match instruction.opcode {
                    op::ACCESS_CHAIN
                    | op::IN_BOUNDS_ACCESS_CHAIN
                    | op::PTR_ACCESS_CHAIN
                    | op::IN_BOUNDS_PTR_ACCESS_CHAIN
                        if operands.len() >= 3 && self.value_carries_provenance(operands[1]) =>
                    {
                        let base = operands[2];
                        let first_index = if matches!(
                            instruction.opcode,
                            op::PTR_ACCESS_CHAIN | op::IN_BOUNDS_PTR_ACCESS_CHAIN
                        ) {
                            4
                        } else {
                            3
                        };
                        let constraints =
                            self.access_chain_constraints(base, operands.get(first_index..).unwrap_or_default());
                        let additions = origins_from(&provenance, [base])
                            .into_iter()
                            .map(|origin| Origin {
                                non_readable: origin.non_readable || constraints.0,
                                non_writable: origin.non_writable || constraints.1,
                                ..origin
                            })
                            .collect();
                        changed |= extend_origins(&mut provenance, operands[1], additions);
                    },
                    op::LOAD
                    | op::IMAGE_TEXEL_POINTER
                    | op::COPY_OBJECT
                    | op::COPY_LOGICAL
                    | op::IMAGE
                    | op::BITCAST
                        if operands.len() >= 3 && self.value_carries_provenance(operands[1]) =>
                    {
                        let additions = origins_from(&provenance, [operands[2]]);
                        changed |= extend_origins(&mut provenance, operands[1], additions);
                    },
                    op::SAMPLED_IMAGE if operands.len() >= 4 && self.value_carries_provenance(operands[1]) => {
                        let additions = origins_from(&provenance, [operands[2], operands[3]]);
                        changed |= extend_origins(&mut provenance, operands[1], additions);
                    },
                    op::SELECT if operands.len() >= 5 && self.value_carries_provenance(operands[1]) => {
                        let additions = origins_from(&provenance, [operands[3], operands[4]]);
                        changed |= extend_origins(&mut provenance, operands[1], additions);
                    },
                    op::PHI if operands.len() >= 4 && self.value_carries_provenance(operands[1]) => {
                        let additions = origins_from(&provenance, operands[2..].iter().step_by(2).copied());
                        changed |= extend_origins(&mut provenance, operands[1], additions);
                    },
                    op::FUNCTION_CALL if operands.len() >= 3 => {
                        let callee = operands[2];
                        if let Some(callee_parameters) = parameters.get(&callee) {
                            for (argument, parameter) in operands[3..].iter().zip(callee_parameters) {
                                if self.value_carries_provenance(*parameter) {
                                    let additions = origins_from(&provenance, [*argument]);
                                    changed |= extend_origins(&mut provenance, *parameter, additions);
                                }
                            }
                        }

                        if self.value_carries_provenance(operands[1]) {
                            let additions =
                                origins_from(&provenance, return_values.get(&callee).into_iter().flatten().copied());
                            changed |= extend_origins(&mut provenance, operands[1], additions);
                        }
                    },
                    _ => {},
                }
            }

            if !changed {
                break;
            }
        }

        let mut access = descriptors
            .iter()
            .map(|descriptor| (descriptor.variable, Access::empty()))
            .collect::<HashMap<_, _>>();
        for instruction in self
            .instructions
            .iter()
            .filter(|instruction| reachable.contains(&instruction.function))
        {
            let operands = instruction.operands.as_slice();
            let mut record = |value, operation| {
                record_descriptor_operation(
                    value,
                    operation,
                    stage,
                    &provenance,
                    &descriptors_by_variable,
                    &mut access,
                )
            };

            match instruction.opcode {
                op::LOAD => {
                    if operands.len() >= 3 && !self.value_is_opaque_handle(operands[1]) {
                        record(operands[2], DescriptorOperation::MemoryRead)?;
                    }
                },
                op::STORE => {
                    if let Some(pointer) = operands.first() {
                        record(*pointer, DescriptorOperation::MemoryWrite)?;
                    }
                },
                op::COPY_MEMORY | op::COPY_MEMORY_SIZED => {
                    if operands.len() >= 2 {
                        record(operands[0], DescriptorOperation::MemoryWrite)?;
                        record(operands[1], DescriptorOperation::MemoryRead)?;
                    }
                },
                op::ARRAY_LENGTH => {
                    if operands.len() >= 3 {
                        record(operands[2], DescriptorOperation::MemoryRead)?;
                    }
                },
                op::ATOMIC_LOAD => {
                    if operands.len() >= 3 {
                        record(operands[2], DescriptorOperation::MemoryRead)?;
                    }
                },
                op::ATOMIC_STORE | op::ATOMIC_FLAG_CLEAR => {
                    if let Some(pointer) = operands.first() {
                        record(*pointer, DescriptorOperation::MemoryWrite)?;
                    }
                },
                op::ATOMIC_RMW_FIRST..=op::ATOMIC_RMW_LAST
                | op::ATOMIC_FLAG_TEST_AND_SET
                | op::ATOMIC_FMIN
                | op::ATOMIC_FMAX
                | op::ATOMIC_FADD => {
                    if operands.len() >= 3 {
                        record(operands[2], DescriptorOperation::MemoryRead)?;
                        record(operands[2], DescriptorOperation::MemoryWrite)?;
                    }
                },
                op::IMAGE_SAMPLE_FIRST..=op::IMAGE_READ
                | op::IMAGE_SPARSE_SAMPLE_FIRST..=op::IMAGE_SPARSE_DREF_GATHER
                | op::IMAGE_SPARSE_READ
                | op::IMAGE_SAMPLE_FOOTPRINT => {
                    if operands.len() >= 3 {
                        record(operands[2], DescriptorOperation::ImageRead)?;
                    }
                },
                op::IMAGE_WRITE => {
                    if let Some(image) = operands.first() {
                        record(*image, DescriptorOperation::ImageWrite)?;
                    }
                },
                op::TRACE_RAY => {
                    if let Some(acceleration_structure) = operands.first() {
                        record(*acceleration_structure, DescriptorOperation::AccelerationStructureRead)?;
                    }
                },
                op::RAY_QUERY_INITIALIZE => {
                    if operands.len() >= 2 {
                        record(operands[1], DescriptorOperation::AccelerationStructureRead)?;
                    }
                },
                op::FUNCTION
                | op::NOP
                | op::LINE
                | op::FUNCTION_PARAMETER
                | op::FUNCTION_END
                | op::FUNCTION_CALL
                | op::VARIABLE
                | op::IMAGE_TEXEL_POINTER
                | op::ACCESS_CHAIN
                | op::IN_BOUNDS_ACCESS_CHAIN
                | op::PTR_ACCESS_CHAIN
                | op::IN_BOUNDS_PTR_ACCESS_CHAIN
                | op::COPY_OBJECT
                | op::COPY_LOGICAL
                | op::SAMPLED_IMAGE
                | op::IMAGE
                | op::BITCAST
                | op::SELECT
                | op::PHI
                | op::RETURN_VALUE
                | op::VECTOR_EXTRACT_DYNAMIC..=op::COMPOSITE_INSERT
                | op::TRANSPOSE
                | op::LOOP_MERGE
                | op::SELECTION_MERGE
                | op::LABEL
                | op::BRANCH
                | op::BRANCH_CONDITIONAL
                | op::SWITCH
                | op::KILL
                | op::RETURN
                | op::UNREACHABLE
                | op::NO_LINE
                | op::IMAGE_QUERY_FORMAT..=op::IMAGE_QUERY_SAMPLES => {},
                op::EXT_INST => {
                    if operands
                        .get(4..)
                        .unwrap_or_default()
                        .iter()
                        .any(|operand| provenance.get(operand).is_some_and(|origins| !origins.is_empty()))
                    {
                        tracing::error!(
                            function = instruction.function,
                            "unsupported extended SPIR-V instruction touches a descriptor"
                        );
                        return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
                    }
                },
                opcode => {
                    if operands
                        .iter()
                        .any(|operand| provenance.get(operand).is_some_and(|origins| !origins.is_empty()))
                    {
                        tracing::error!(
                            opcode,
                            function = instruction.function,
                            "unsupported SPIR-V instruction touches a descriptor"
                        );
                        return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
                    }
                },
            }
        }

        for descriptor_access in access.values_mut() {
            if descriptor_access.is_empty() {
                *descriptor_access = Access::None;
            }
        }
        Ok(access)
    }
}

fn record_descriptor_operation(
    value: u32, operation: DescriptorOperation, stage: vk::ShaderStageFlags, provenance: &Provenance,
    descriptors: &HashMap<u32, &PendingDescriptor>, accesses: &mut HashMap<u32, Access>,
) -> Result<(), vk::Result> {
    let Some(origins) = provenance.get(&value) else {
        return Ok(());
    };

    let reads = !matches!(
        operation,
        DescriptorOperation::MemoryWrite | DescriptorOperation::ImageWrite
    );
    let writes = matches!(
        operation,
        DescriptorOperation::MemoryWrite | DescriptorOperation::ImageWrite
    );
    for origin in origins {
        let descriptor = descriptors
            .get(&origin.variable)
            .expect("descriptor provenance must name a reflected descriptor");
        if (reads && origin.non_readable) || (writes && origin.non_writable) {
            tracing::error!(
                variable = origin.variable,
                ?operation,
                non_readable = origin.non_readable,
                non_writable = origin.non_writable,
                "SPIR-V descriptor operation contradicts its access decorations"
            );
            return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
        }

        let descriptor_type = descriptor.binding.descriptor_type;
        let operation_access = match operation {
            DescriptorOperation::MemoryRead => match descriptor_type {
                vk::DescriptorType::UNIFORM_BUFFER => Access::uniform_by(stage),
                vk::DescriptorType::STORAGE_BUFFER
                | vk::DescriptorType::STORAGE_IMAGE
                | vk::DescriptorType::STORAGE_TEXEL_BUFFER => Access::shader_read_by(stage),
                _ => {
                    tracing::error!(
                        variable = origin.variable,
                        ?descriptor_type,
                        ?operation,
                        "SPIR-V memory operation is incompatible with its descriptor type"
                    );
                    return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
                },
            },
            DescriptorOperation::MemoryWrite => match descriptor_type {
                vk::DescriptorType::STORAGE_BUFFER
                | vk::DescriptorType::STORAGE_IMAGE
                | vk::DescriptorType::STORAGE_TEXEL_BUFFER => Access::shader_write_by(stage),
                _ => {
                    tracing::error!(
                        variable = origin.variable,
                        ?descriptor_type,
                        ?operation,
                        "SPIR-V memory operation is incompatible with its descriptor type"
                    );
                    return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
                },
            },
            DescriptorOperation::ImageRead => match descriptor_type {
                vk::DescriptorType::SAMPLER => Access::empty(),
                vk::DescriptorType::SAMPLED_IMAGE
                | vk::DescriptorType::COMBINED_IMAGE_SAMPLER
                | vk::DescriptorType::UNIFORM_TEXEL_BUFFER => Access::sampled_by(stage),
                vk::DescriptorType::STORAGE_IMAGE | vk::DescriptorType::STORAGE_TEXEL_BUFFER => {
                    Access::shader_read_by(stage)
                },
                vk::DescriptorType::INPUT_ATTACHMENT => Access::InputAttachmentRead,
                _ => {
                    tracing::error!(
                        variable = origin.variable,
                        ?descriptor_type,
                        ?operation,
                        "SPIR-V image operation is incompatible with its descriptor type"
                    );
                    return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
                },
            },
            DescriptorOperation::ImageWrite => match descriptor_type {
                vk::DescriptorType::SAMPLER => Access::empty(),
                vk::DescriptorType::STORAGE_IMAGE | vk::DescriptorType::STORAGE_TEXEL_BUFFER => {
                    Access::shader_write_by(stage)
                },
                _ => {
                    tracing::error!(
                        variable = origin.variable,
                        ?descriptor_type,
                        ?operation,
                        "SPIR-V image operation is incompatible with its descriptor type"
                    );
                    return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
                },
            },
            DescriptorOperation::AccelerationStructureRead => match descriptor_type {
                vk::DescriptorType::ACCELERATION_STRUCTURE_KHR => Access::acceleration_structure_read_by(stage),
                _ => {
                    tracing::error!(
                        variable = origin.variable,
                        ?descriptor_type,
                        ?operation,
                        "SPIR-V acceleration-structure operation is incompatible with its descriptor type"
                    );
                    return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
                },
            },
        };

        *accesses
            .get_mut(&origin.variable)
            .expect("every reflected descriptor must have an access entry") |= operation_access;
    }

    Ok(())
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

    let [entry_point] = parsed.entry_points.as_slice() else {
        tracing::error!(
            count = parsed.entry_points.len(),
            "shader must declare exactly one entry point"
        );
        return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
    };

    let Some(stage) = stage_from_execution_model(entry_point.execution_model) else {
        tracing::error!(
            execution_model = entry_point.execution_model,
            "unsupported SPIR-V execution model"
        );
        return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
    };

    let mut pending_descriptors = Vec::new();
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

        pending_descriptors.push(PendingDescriptor {
            variable: *variable,
            pointer_type: *pointer_type,
            binding: DescriptorBinding {
                set,
                binding,
                descriptor_type,
                count,
                variable_count,
                stages: stage,
                access: Access::None,
            },
        });
    }

    let descriptor_accesses = parsed.descriptor_accesses(entry_point.function, stage, &pending_descriptors)?;
    let mut bindings = pending_descriptors
        .into_iter()
        .map(|mut descriptor| {
            descriptor.binding.access = descriptor_accesses[&descriptor.variable];
            descriptor.binding
        })
        .collect::<Vec<_>>();

    bindings.sort_unstable_by_key(|b| (b.set, b.binding));
    vertex_inputs.sort_unstable_by_key(|input| input.location);

    let (push_constant_offset, push_constant_size) = match push_constants {
        Some((start, end)) => (start, end - start),
        None => (0, 0),
    };

    let local_size = match parsed.declared_local_size() {
        Some(local_size) if local_size.iter().all(|axis| *axis > 0) => local_size,
        _ => {
            if stage == vk::ShaderStageFlags::COMPUTE {
                tracing::warn!("compute shader declares no workgroup size; it is taken to be one invocation");
            }
            [1, 1, 1]
        },
    };

    Ok(Reflection {
        stage,
        entry_point: entry_point.name.clone(),
        bindings,
        push_constant_offset,
        push_constant_size,
        vertex_inputs,
        local_size,
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

    /// A compute module whose workgroup is spelled the way `[numthreads]` compiles down to.
    fn compute_module(mode: u16, local_size: [u32; 3]) -> Vec<u32> {
        let mut words = vec![MAGIC, 0x0001_0300, 0, 100, 0];

        let mut entry = vec![5, 99];
        entry.extend(literal("cs_main"));
        words.extend(inst(op::ENTRY_POINT, &entry));

        let operands = match mode {
            op::EXECUTION_MODE_ID => [99, execution_mode::LOCAL_SIZE_ID, 30, 31, 32],
            _ => [
                99,
                execution_mode::LOCAL_SIZE,
                local_size[0],
                local_size[1],
                local_size[2],
            ],
        };
        words.extend(inst(mode, &operands));

        words.extend(inst(op::TYPE_INT, &[11, 32, 0]));
        for (id, axis) in [30, 31, 32].into_iter().zip(local_size) {
            words.extend(inst(op::CONSTANT, &[11, id, axis]));
        }

        words
    }

    #[derive(Clone, Copy)]
    enum BufferOperation {
        None,
        Read,
        Write,
        ReadWrite,
        Unknown,
    }

    fn buffer_descriptor_module(
        storage_class: u32, operation: BufferOperation, member_decoration: Option<u32>,
    ) -> Vec<u32> {
        let mut words = vec![MAGIC, 0x0001_0300, 0, 100, 0];

        let mut entry = vec![5, 9];
        entry.extend(literal("main"));
        words.extend(inst(op::ENTRY_POINT, &entry));
        words.extend(inst(op::EXECUTION_MODE, &[9, execution_mode::LOCAL_SIZE, 1, 1, 1]));
        words.extend(inst(op::DECORATE, &[6, decoration::DESCRIPTOR_SET, 0]));
        words.extend(inst(op::DECORATE, &[6, decoration::BINDING, 0]));
        if let Some(decoration) = member_decoration {
            words.extend(inst(op::MEMBER_DECORATE, &[4, 0, decoration]));
        }

        words.extend(inst(op::TYPE_VOID, &[1]));
        words.extend(inst(op::TYPE_FUNCTION, &[2, 1]));
        words.extend(inst(op::TYPE_INT, &[3, 32, 0]));
        words.extend(inst(op::TYPE_STRUCT, &[4, 3]));
        words.extend(inst(op::TYPE_POINTER, &[5, storage_class, 4]));
        words.extend(inst(op::VARIABLE, &[5, 6, storage_class]));
        words.extend(inst(op::TYPE_POINTER, &[7, storage_class, 3]));
        words.extend(inst(op::CONSTANT, &[3, 8, 0]));

        words.extend(inst(op::FUNCTION, &[1, 9, 0, 2]));
        words.extend(inst(op::LABEL, &[20]));
        words.extend(inst(op::ACCESS_CHAIN, &[7, 10, 6, 8]));
        if matches!(operation, BufferOperation::Read | BufferOperation::ReadWrite) {
            words.extend(inst(op::LOAD, &[3, 11, 10]));
        }
        if matches!(operation, BufferOperation::Write | BufferOperation::ReadWrite) {
            words.extend(inst(op::STORE, &[10, 8]));
        }
        if matches!(operation, BufferOperation::Unknown) {
            words.extend(inst(999, &[10]));
        }
        words.extend(inst(op::RETURN, &[]));
        words.extend(inst(op::FUNCTION_END, &[]));
        words
    }

    fn buffer_descriptor_through_function(call_helper: bool) -> Vec<u32> {
        let mut words = vec![MAGIC, 0x0001_0300, 0, 100, 0];

        let mut entry = vec![5, 9];
        entry.extend(literal("main"));
        words.extend(inst(op::ENTRY_POINT, &entry));
        words.extend(inst(op::EXECUTION_MODE, &[9, execution_mode::LOCAL_SIZE, 1, 1, 1]));
        words.extend(inst(op::DECORATE, &[6, decoration::DESCRIPTOR_SET, 0]));
        words.extend(inst(op::DECORATE, &[6, decoration::BINDING, 0]));

        words.extend(inst(op::TYPE_VOID, &[1]));
        words.extend(inst(op::TYPE_FUNCTION, &[2, 1]));
        words.extend(inst(op::TYPE_INT, &[3, 32, 0]));
        words.extend(inst(op::TYPE_STRUCT, &[4, 3]));
        words.extend(inst(op::TYPE_POINTER, &[5, storage_class::STORAGE_BUFFER, 4]));
        words.extend(inst(op::VARIABLE, &[5, 6, storage_class::STORAGE_BUFFER]));
        words.extend(inst(op::TYPE_POINTER, &[7, storage_class::STORAGE_BUFFER, 3]));
        words.extend(inst(op::CONSTANT, &[3, 8, 0]));
        words.extend(inst(op::TYPE_FUNCTION, &[13, 1, 7]));

        words.extend(inst(op::FUNCTION, &[1, 9, 0, 2]));
        words.extend(inst(op::LABEL, &[20]));
        if call_helper {
            words.extend(inst(op::ACCESS_CHAIN, &[7, 10, 6, 8]));
            words.extend(inst(op::FUNCTION_CALL, &[1, 11, 12, 10]));
        }
        words.extend(inst(op::RETURN, &[]));
        words.extend(inst(op::FUNCTION_END, &[]));

        words.extend(inst(op::FUNCTION, &[1, 12, 0, 13]));
        words.extend(inst(op::FUNCTION_PARAMETER, &[7, 14]));
        words.extend(inst(op::LABEL, &[21]));
        words.extend(inst(op::STORE, &[14, 8]));
        words.extend(inst(op::RETURN, &[]));
        words.extend(inst(op::FUNCTION_END, &[]));
        words
    }

    #[derive(Clone, Copy)]
    enum ImageOperation {
        None,
        Read,
        Write,
    }

    fn image_descriptor_module(
        execution_model: u32, image_dim: u32, sampled: u32, operation: ImageOperation,
    ) -> Vec<u32> {
        let mut words = vec![MAGIC, 0x0001_0300, 0, 100, 0];

        let mut entry = vec![execution_model, 9];
        entry.extend(literal("main"));
        words.extend(inst(op::ENTRY_POINT, &entry));
        if execution_model == 5 {
            words.extend(inst(op::EXECUTION_MODE, &[9, execution_mode::LOCAL_SIZE, 1, 1, 1]));
        }
        words.extend(inst(op::DECORATE, &[6, decoration::DESCRIPTOR_SET, 0]));
        words.extend(inst(op::DECORATE, &[6, decoration::BINDING, 0]));

        words.extend(inst(op::TYPE_VOID, &[1]));
        words.extend(inst(op::TYPE_FUNCTION, &[2, 1]));
        words.extend(inst(op::TYPE_FLOAT, &[3, 32]));
        words.extend(inst(op::TYPE_IMAGE, &[4, 3, image_dim, 0, 0, 0, sampled, 0]));
        words.extend(inst(op::TYPE_POINTER, &[5, storage_class::UNIFORM_CONSTANT, 4]));
        words.extend(inst(op::VARIABLE, &[5, 6, storage_class::UNIFORM_CONSTANT]));
        words.extend(inst(op::CONSTANT, &[3, 8, 0]));

        words.extend(inst(op::FUNCTION, &[1, 9, 0, 2]));
        words.extend(inst(op::LABEL, &[20]));
        words.extend(inst(op::LOAD, &[4, 10, 6]));
        match operation {
            ImageOperation::None => {},
            ImageOperation::Read => words.extend(inst(op::IMAGE_READ, &[3, 11, 10, 8])),
            ImageOperation::Write => words.extend(inst(op::IMAGE_WRITE, &[10, 8, 8])),
        }
        words.extend(inst(op::RETURN, &[]));
        words.extend(inst(op::FUNCTION_END, &[]));
        words
    }

    fn acceleration_structure_module() -> Vec<u32> {
        let mut words = vec![MAGIC, 0x0001_0300, 0, 100, 0];

        let mut entry = vec![5, 9];
        entry.extend(literal("main"));
        words.extend(inst(op::ENTRY_POINT, &entry));
        words.extend(inst(op::EXECUTION_MODE, &[9, execution_mode::LOCAL_SIZE, 1, 1, 1]));
        words.extend(inst(op::DECORATE, &[6, decoration::DESCRIPTOR_SET, 0]));
        words.extend(inst(op::DECORATE, &[6, decoration::BINDING, 0]));

        words.extend(inst(op::TYPE_VOID, &[1]));
        words.extend(inst(op::TYPE_FUNCTION, &[2, 1]));
        words.extend(inst(op::TYPE_ACCELERATION_STRUCTURE, &[4]));
        words.extend(inst(op::TYPE_POINTER, &[5, storage_class::UNIFORM_CONSTANT, 4]));
        words.extend(inst(op::VARIABLE, &[5, 6, storage_class::UNIFORM_CONSTANT]));

        words.extend(inst(op::FUNCTION, &[1, 9, 0, 2]));
        words.extend(inst(op::LABEL, &[20]));
        words.extend(inst(op::LOAD, &[4, 10, 6]));
        words.extend(inst(op::RAY_QUERY_INITIALIZE, &[30, 10, 31, 32, 33, 34, 35, 36]));
        words.extend(inst(op::RETURN, &[]));
        words.extend(inst(op::FUNCTION_END, &[]));
        words
    }

    fn only_binding_access(module: &[u32]) -> Access {
        let reflection = reflect(module).expect("module should reflect");
        assert_eq!(reflection.bindings.len(), 1);
        assert_eq!(reflection.bindings[0].stages, vk::ShaderStageFlags::COMPUTE);
        reflection.bindings[0].access
    }

    #[test]
    fn a_compute_uniform_buffer_read_has_uniform_access_at_compute() {
        assert_eq!(
            only_binding_access(&buffer_descriptor_module(
                storage_class::UNIFORM,
                BufferOperation::Read,
                None,
            )),
            Access::ComputeUniformRead
        );
    }

    #[test]
    fn compute_storage_buffer_access_distinguishes_reads_and_writes() {
        assert_eq!(
            only_binding_access(&buffer_descriptor_module(
                storage_class::STORAGE_BUFFER,
                BufferOperation::Read,
                None,
            )),
            Access::ComputeRead
        );
        assert_eq!(
            only_binding_access(&buffer_descriptor_module(
                storage_class::STORAGE_BUFFER,
                BufferOperation::Write,
                None,
            )),
            Access::ComputeWrite
        );
        assert_eq!(
            only_binding_access(&buffer_descriptor_module(
                storage_class::STORAGE_BUFFER,
                BufferOperation::ReadWrite,
                None,
            )),
            Access::ComputeRW
        );
    }

    #[test]
    fn compute_storage_images_distinguish_reads_and_writes() {
        assert_eq!(
            only_binding_access(&image_descriptor_module(5, 1, 2, ImageOperation::Read)),
            Access::ComputeRead
        );
        assert_eq!(
            only_binding_access(&image_descriptor_module(5, 1, 2, ImageOperation::Write)),
            Access::ComputeWrite
        );
    }

    #[test]
    fn loading_an_image_handle_alone_does_not_access_the_image() {
        assert_eq!(
            only_binding_access(&image_descriptor_module(5, 1, 1, ImageOperation::None)),
            Access::None
        );
    }

    #[test]
    fn a_compute_acceleration_structure_query_has_stage_specific_access() {
        assert_eq!(
            only_binding_access(&acceleration_structure_module()),
            Access::ComputeAccelerationStructureRead
        );
    }

    #[test]
    fn a_fragment_subpass_read_has_input_attachment_access() {
        let reflection = reflect(&image_descriptor_module(4, dim::SUBPASS_DATA, 2, ImageOperation::Read))
            .expect("module should reflect");
        assert_eq!(
            reflection.bindings[0].descriptor_type,
            vk::DescriptorType::INPUT_ATTACHMENT
        );
        assert_eq!(reflection.bindings[0].access, Access::InputAttachmentRead);
    }

    #[test]
    fn declaring_or_aliasing_a_descriptor_does_not_access_it() {
        assert_eq!(
            only_binding_access(&buffer_descriptor_module(
                storage_class::STORAGE_BUFFER,
                BufferOperation::None,
                None,
            )),
            Access::None
        );
    }

    #[test]
    fn descriptor_access_follows_the_static_function_call_tree() {
        assert_eq!(
            only_binding_access(&buffer_descriptor_through_function(true)),
            Access::ComputeWrite
        );
        assert_eq!(
            only_binding_access(&buffer_descriptor_through_function(false)),
            Access::None
        );
    }

    #[test]
    fn descriptor_operations_must_respect_member_access_decorations() {
        assert!(
            reflect(&buffer_descriptor_module(
                storage_class::STORAGE_BUFFER,
                BufferOperation::Write,
                Some(decoration::NON_WRITABLE),
            ))
            .is_err()
        );
        assert!(
            reflect(&buffer_descriptor_module(
                storage_class::STORAGE_BUFFER,
                BufferOperation::Read,
                Some(decoration::NON_READABLE),
            ))
            .is_err()
        );
    }

    #[test]
    fn rejects_an_unknown_instruction_that_touches_a_descriptor() {
        assert!(
            reflect(&buffer_descriptor_module(
                storage_class::STORAGE_BUFFER,
                BufferOperation::Unknown,
                None,
            ))
            .is_err()
        );
    }

    #[test]
    fn reads_the_workgroup_a_compute_shader_declares() {
        let reflection = reflect(&compute_module(op::EXECUTION_MODE, [64, 2, 1])).expect("module should reflect");
        assert_eq!(reflection.stage, vk::ShaderStageFlags::COMPUTE);
        assert_eq!(reflection.local_size, [64, 2, 1]);
    }

    #[test]
    fn reads_a_workgroup_that_names_constants_rather_than_literals() {
        let reflection = reflect(&compute_module(op::EXECUTION_MODE_ID, [8, 8, 4])).expect("module should reflect");
        assert_eq!(reflection.local_size, [8, 8, 4]);
    }

    #[test]
    fn a_stage_with_no_workgroup_reports_one_invocation() {
        let reflection = reflect(&vertex_module()).expect("module should reflect");
        assert_eq!(reflection.local_size, [1, 1, 1]);
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
