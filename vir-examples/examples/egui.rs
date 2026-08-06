//! The overlay on its own.
//!
//! Every example draws its controls through the same egui backend the harness owns, so this one
//! is what that backend looks like with nothing underneath it: the frame clears the swapchain
//! and hands it straight back, and the entire image is the UI. The backend itself lives in
//! `src/egui_pass.rs`, and between the font atlas and the meshes it is where texture uploads,
//! indexed draws, per-draw scissors and bindless sampling are exercised.

use ash::vk;
use vir::{ClearValue, ValueId};
use vir_examples::{Example, Frame, Setup, egui};

struct Egui {
    background: [f32; 3],
    text: String,
    counter: i32,
}

impl Default for Egui {
    fn default() -> Self {
        Self {
            background: [0.02, 0.02, 0.05],
            text: String::from("the font atlas is a sampled texture, so this text is the upload path"),
            counter: 0,
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
                    *self = Self::default();
                }
            });
    }

    fn render(&mut self, frame: &mut Frame) -> Result<ValueId, vk::Result> {
        let [r, g, b] = self.background;

        Ok(frame
            .module
            .clear(frame.swapchain_image, ClearValue::rgba_f32(r, g, b, 1.0)))
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> { vir_examples::run::<Egui>() }
