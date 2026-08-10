use std::{fmt, sync::Arc};

use crate::graph::render_graph::Recorder;

#[derive(Clone)]
pub struct PassCallback(Arc<dyn Fn(&mut Recorder<'_>) + Send + Sync>);

impl PassCallback {
    pub fn new(body: impl Fn(&mut Recorder<'_>) + Send + Sync + 'static) -> Self { Self(Arc::new(body)) }

    pub fn empty() -> Self { Self::new(|_| {}) }

    pub(crate) fn call(&self, recorder: &mut Recorder<'_>) { (self.0)(recorder) }
}

impl fmt::Debug for PassCallback {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str("callback") }
}
