//! The egui overlay on its own.
//!
//! Every example draws its controls through the same egui backend the harness owns. Here there
//! is nothing under it: the frame clears the swapchain and hands it back, and the whole image is
//! the UI. The backend lives in `src/egui_pass.rs`, where the font atlas and meshes exercise
//! texture uploads, indexed draws, per-draw scissors, and ordinary descriptor binding.

use ash::vk;
use vir::{ClearValue, ValueId};
use vir_examples::{Example, Frame, Recording, Setup, egui};

struct Egui {
    background: [f32; 3],
    text: String,
    counter: i32,
    /// The colour the compiled clear reads, which is the whole of what this example rebuilds
    /// per frame.
    clear_color: Option<ValueId>,
}

impl Default for Egui {
    fn default() -> Self {
        Self {
            background: [0.02, 0.02, 0.05],
            text: String::from("the font atlas is a sampled texture, so this text is the upload path"),
            counter: 0,
            clear_color: None,
        }
    }
}

impl Example for Egui {
    const TITLE: &'static str = "vir: egui";

    fn new(_setup: &mut Setup) -> Result<Self, vk::Result> { Ok(Self::default()) }

    fn ui(&mut self, ctx: &egui::Context) {
        egui::Window::new(Self::TITLE)
            .default_pos([24.0, 24.0])
            .default_size([360.0, 260.0])
            .show(ctx, |ui| {
                ui.label("nothing is drawn under this window; the clear is all there is");
                ui.separator();

                ui.color_edit_button_rgb(&mut self.background);
                ui.label("background");
                ui.separator();

                ui.text_edit_multiline(&mut self.text);
                ui.separator();

                ui.horizontal(|ui| {
                    if ui.button("-").clicked() {
                        self.counter -= 1;
                    }
                    ui.label(self.counter.to_string());
                    if ui.button("+").clicked() {
                        self.counter += 1;
                    }
                });

                ui.separator();
                if ui.button("reset").clicked() {
                    let clear_color = self.clear_color;
                    *self = Self {
                        clear_color,
                        ..Self::default()
                    };
                }
            });
    }

    fn record(&mut self, recording: &mut Recording) -> Result<ValueId, vk::Result> {
        let clear_color = recording.module.declare_clear_var("background", vir::clear::f32::BLACK);
        self.clear_color = Some(clear_color);

        Ok(recording.module.clear_from(recording.swapchain_image, clear_color))
    }

    fn update(&mut self, frame: &mut Frame) -> Result<(), vk::Result> {
        let [r, g, b] = self.background;
        if let Some(clear_color) = self.clear_color {
            frame.program.set(clear_color, ClearValue::rgba_f32(r, g, b, 1.0));
        }

        Ok(())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> { vir_examples::run::<Egui>() }
