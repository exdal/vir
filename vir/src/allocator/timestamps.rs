use std::{ptr::NonNull, sync::Arc, time::Duration};

use ash::vk;

use crate::CommandBuffer;

const QUERY_CAPACITY: u32 = 512;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PassTiming {
    pub name: Arc<str>,
    pub duration: Duration,
}

#[derive(Debug)]
pub(crate) struct TimestampPool {
    device: NonNull<ash::Device>,
    pool: vk::QueryPool,
    period_ns: f64,
    valid_mask: u64,
    is_reset: bool,
    passes: Vec<Arc<str>>,
    completed: Vec<PassTiming>,
}

impl TimestampPool {
    pub(crate) fn new(device: NonNull<ash::Device>, period_ns: f32, valid_bits: u32) -> Result<Self, vk::Result> {
        let create_info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(QUERY_CAPACITY);
        let pool = unsafe { device.as_ref().create_query_pool(&create_info, None) }?;
        let valid_mask = match valid_bits {
            64.. => u64::MAX,
            bits => (1 << bits) - 1,
        };

        Ok(Self {
            device,
            pool,
            period_ns: f64::from(period_ns),
            valid_mask,
            is_reset: false,
            passes: Vec::new(),
            completed: Vec::new(),
        })
    }

    /// Records outside a render pass instance, since the first pass of a frame resets the pool.
    pub(crate) fn begin(&mut self, cmd: &CommandBuffer, name: Arc<str>) -> Option<u32> {
        let query = u32::try_from(self.passes.len()).ok()?.checked_mul(2)?;
        if query + 2 > QUERY_CAPACITY {
            return None;
        }

        if !self.is_reset {
            cmd.reset_query_pool(self.pool, 0, QUERY_CAPACITY);
            self.is_reset = true;
        }

        cmd.write_timestamp(vk::PipelineStageFlags2::TOP_OF_PIPE, self.pool, query);
        self.passes.push(name);

        Some(query + 1)
    }

    pub(crate) fn end(&self, cmd: &CommandBuffer, query: u32) {
        cmd.write_timestamp(vk::PipelineStageFlags2::ALL_COMMANDS, self.pool, query);
    }

    /// The caller has waited for every submission that wrote to the pool.
    pub(crate) fn resolve(&mut self) {
        self.completed.clear();
        self.is_reset = false;
        let passes = std::mem::take(&mut self.passes);
        if passes.is_empty() {
            return;
        }

        let mut ticks = vec![0u64; passes.len() * 2];
        let result = unsafe {
            self.device
                .as_ref()
                .get_query_pool_results(self.pool, 0, &mut ticks, vk::QueryResultFlags::TYPE_64)
        };
        // a frame that failed before submitting never wrote its queries
        if result.is_err() {
            return;
        }

        self.completed = passes
            .into_iter()
            .zip(ticks.as_chunks::<2>().0)
            .map(|(name, [begin, end])| {
                let elapsed = end.wrapping_sub(*begin) & self.valid_mask;
                let nanos = (elapsed as f64 * self.period_ns) as u64;

                PassTiming {
                    name,
                    duration: Duration::from_nanos(nanos),
                }
            })
            .collect();
    }

    pub(crate) fn completed(&self) -> &[PassTiming] { &self.completed }
}

impl Drop for TimestampPool {
    fn drop(&mut self) { unsafe { self.device.as_ref().destroy_query_pool(self.pool, None) } }
}
