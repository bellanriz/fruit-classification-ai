import os
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD

    TKDND_AVAILABLE = True
except Exception:
    TKDND_AVAILABLE = False


MODEL_PATH = "models/fruit_classifier.h5"
CLASS_INDICES_PATH = "models/class_indices.npy"
IMG_SIZE = (100, 100)

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

THEMES = {
    "Fresh": {
        "bg": "#f4f7fb",
        "surface": "#ffffff",
        "header": "#1f9d8b",
        "header_text": "#ffffff",
        "header_sub": "#e8f8f5",
        "text": "#1d2a38",
        "muted": "#5e6c7a",
        "accent": "#1f9d8b",
        "accent_dark": "#177565",
        "button_secondary": "#2d3e50",
        "button_secondary_dark": "#223243",
        "warn": "#d94841",
        "preview_bg": "#f0f6ff",
        "preview_border": "#c7d7ea",
        "card_border": "#d8e0ea",
        "meter_trough": "#d7e5ef",
        "score_bar": "#23b49c",
        "chip_bg": "#e7f7f3",
        "chip_text": "#176357",
        "confidence_high": "#118a5a",
        "confidence_mid": "#c67a00",
        "confidence_low": "#c53f3a",
    },
    "Citrus": {
        "bg": "#fff8ef",
        "surface": "#ffffff",
        "header": "#e9752e",
        "header_text": "#ffffff",
        "header_sub": "#ffefe3",
        "text": "#2f2a22",
        "muted": "#7a6553",
        "accent": "#e9752e",
        "accent_dark": "#b9571a",
        "button_secondary": "#45617d",
        "button_secondary_dark": "#344a61",
        "warn": "#d94841",
        "preview_bg": "#fff4e8",
        "preview_border": "#efd3b6",
        "card_border": "#eddac8",
        "meter_trough": "#ecd7c3",
        "score_bar": "#f39b2f",
        "chip_bg": "#ffeedb",
        "chip_text": "#8b4f19",
        "confidence_high": "#237a47",
        "confidence_mid": "#b46812",
        "confidence_low": "#c53f3a",
    },
}


def load_class_names(class_indices_path: str) -> list[str]:
    class_indices = np.load(class_indices_path, allow_pickle=True).item()
    index_to_class = {idx: name for name, idx in class_indices.items()}
    return [index_to_class[i] for i in range(len(index_to_class))]


class FruitClassifierGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Fruit Classification Studio")
        self.root.geometry("980x680")
        self.root.minsize(760, 620)

        if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_INDICES_PATH):
            messagebox.showerror(
                "Missing files",
                "Model or class index files are missing. Run train.py first.",
            )
            self.root.destroy()
            return

        self.model = load_model(MODEL_PATH)
        self.class_names = load_class_names(CLASS_INDICES_PATH)
        self.selected_image_path: str | None = None
        self.image_preview: ImageTk.PhotoImage | None = None
        self.class_widgets: dict[str, tuple[tk.Frame, tk.Label, tk.Label, ttk.Progressbar]] = {}
        self.history_entries: list[str] = []
        self.max_history = 5
        self.current_confidence = 0.0
        self.current_meter_color = THEMES["Fresh"]["accent"]
        self.current_status_color = THEMES["Fresh"]["muted"]
        self.compact_mode = False

        self.theme_name = "Fresh"
        self.theme_var = tk.StringVar(value=self.theme_name)

        self._configure_styles()
        self._build_ui()
        self._apply_theme(self.theme_name)
        self._set_confidence_meter(0.0)
        self._reset_scores()
        self._update_top3([])
        self.root.bind("<Configure>", self._on_window_resize)

    @property
    def theme(self) -> dict[str, str]:
        return THEMES[self.theme_name]

    def _configure_styles(self) -> None:
        self.style = ttk.Style(self.root)
        self.style.theme_use("clam")

    def _set_progress_style(self) -> None:
        self.style.configure(
            "Score.Horizontal.TProgressbar",
            thickness=12,
            troughcolor=self.theme["meter_trough"],
            background=self.theme["score_bar"],
            bordercolor=self.theme["meter_trough"],
            lightcolor=self.theme["score_bar"],
            darkcolor=self.theme["score_bar"],
        )

    def _build_ui(self) -> None:
        self.header = tk.Frame(self.root, height=108)
        self.header.pack(fill=tk.X)
        self.header.pack_propagate(False)

        self.title_label = tk.Label(
            self.header,
            text="Fruit Classification Studio",
            font=("Bahnschrift", 26, "bold"),
        )
        self.title_label.pack(anchor="w", padx=28, pady=(20, 2))

        self.subtitle_label = tk.Label(
            self.header,
            text="Upload a fruit image and inspect prediction confidence in real time.",
            font=("Segoe UI", 11),
        )
        self.subtitle_label.pack(anchor="w", padx=28)

        theme_row = tk.Frame(self.header)
        theme_row.pack(anchor="e", padx=24, pady=(0, 8))

        self.theme_label = tk.Label(
            theme_row,
            text="Theme:",
            font=("Segoe UI", 9, "bold"),
        )
        self.theme_label.pack(side=tk.LEFT, padx=(0, 8))

        self.theme_menu = ttk.OptionMenu(
            theme_row,
            self.theme_var,
            self.theme_name,
            *THEMES.keys(),
            command=self._on_theme_change,
        )
        self.theme_menu.pack(side=tk.LEFT)

        self.body = tk.Frame(self.root)
        self.body.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        self.body.grid_columnconfigure(0, weight=3)
        self.body.grid_columnconfigure(1, weight=2)
        self.body.grid_rowconfigure(0, weight=1)

        self.left_panel = tk.Frame(self.body)
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

        self.right_panel = tk.Frame(self.body)
        self.right_panel.grid(row=0, column=1, sticky="nsew")

        self._build_preview_card(self.left_panel)
        self._build_result_card(self.left_panel)
        self._build_scores_card(self.right_panel)
        self._build_history_card(self.right_panel)
        self._configure_drag_and_drop()
        self._apply_layout_mode(compact=False)

    def _build_preview_card(self, parent: tk.Frame) -> None:
        self.preview_card = tk.Frame(parent, highlightthickness=1)
        self.preview_card.pack(fill=tk.BOTH, expand=True)

        self.preview_title = tk.Label(
            self.preview_card,
            text="Image Preview",
            font=("Segoe UI", 13, "bold"),
        )
        self.preview_title.pack(anchor="w", padx=16, pady=(14, 6))

        self.preview_frame = tk.Frame(
            self.preview_card,
            width=400,
            height=340,
            highlightthickness=1,
        )
        self.preview_frame.pack(padx=16, pady=(4, 12))
        self.preview_frame.pack_propagate(False)

        self.image_label = tk.Label(
            self.preview_frame,
            text="No image selected",
            font=("Segoe UI", 11),
        )
        self.image_label.pack(expand=True)

        self.drop_hint = tk.Label(
            self.preview_card,
            text="Drop an image here or use Upload Image",
            font=("Segoe UI", 9),
        )
        self.drop_hint.pack(anchor="w", padx=16, pady=(0, 10))

        button_row = tk.Frame(self.preview_card)
        button_row.pack(fill=tk.X, padx=16, pady=(0, 6))

        self.upload_button = tk.Button(
            button_row,
            text="Upload Image",
            font=("Segoe UI", 10, "bold"),
            width=15,
            relief=tk.FLAT,
            bd=0,
            cursor="hand2",
            command=self.upload_image,
        )
        self.upload_button.pack(side=tk.LEFT)

        self.predict_button = tk.Button(
            button_row,
            text="Predict",
            font=("Segoe UI", 10, "bold"),
            width=15,
            relief=tk.FLAT,
            bd=0,
            cursor="hand2",
            command=self.predict_image,
            state=tk.DISABLED,
        )
        self.predict_button.pack(side=tk.LEFT, padx=(10, 0))

        self.path_label = tk.Label(
            self.preview_card,
            text="Image path: -",
            font=("Consolas", 9),
            anchor="w",
        )
        self.path_label.pack(fill=tk.X, padx=16, pady=(2, 14))

        self._bind_hover(self.upload_button, "accent")
        self._bind_hover(self.predict_button, "button_secondary")

    def _build_result_card(self, parent: tk.Frame) -> None:
        self.result_card = tk.Frame(parent, highlightthickness=1)
        self.result_card.pack(fill=tk.X, pady=(12, 0))

        self.result_label = tk.Label(
            self.result_card,
            text="Prediction: -",
            font=("Segoe UI", 16, "bold"),
            pady=10,
        )
        self.result_label.pack(anchor="w", padx=16)

        self.confidence_label = tk.Label(
            self.result_card,
            text="Confidence: -",
            font=("Segoe UI", 11),
        )
        self.confidence_label.pack(anchor="w", padx=16)

        self.confidence_meter = tk.Canvas(
            self.result_card,
            width=360,
            height=20,
            highlightthickness=0,
        )
        self.confidence_meter.pack(anchor="w", padx=16, pady=(8, 6))

        self.top3_title = tk.Label(
            self.result_card,
            text="Top-3 Predictions",
            font=("Segoe UI", 10, "bold"),
        )
        self.top3_title.pack(anchor="w", padx=16, pady=(6, 2))

        self.top3_frame = tk.Frame(self.result_card)
        self.top3_frame.pack(fill=tk.X, padx=16, pady=(0, 6))

        self.top3_labels: list[tk.Label] = []
        for _ in range(3):
            chip = tk.Label(
                self.top3_frame,
                text="-",
                font=("Segoe UI", 9, "bold"),
                padx=10,
                pady=4,
                bd=0,
                relief=tk.FLAT,
            )
            chip.pack(side=tk.LEFT, padx=(0, 8))
            self.top3_labels.append(chip)

        self.status_label = tk.Label(
            self.result_card,
            text="Status: Waiting for image upload",
            font=("Segoe UI", 10),
            pady=10,
        )
        self.status_label.pack(anchor="w", padx=16)

    def _build_scores_card(self, parent: tk.Frame) -> None:
        self.scores_card = tk.Frame(parent, highlightthickness=1)
        self.scores_card.pack(fill=tk.BOTH, expand=True)

        self.scores_title = tk.Label(
            self.scores_card,
            text="Class Confidence Breakdown",
            font=("Segoe UI", 13, "bold"),
        )
        self.scores_title.pack(anchor="w", padx=16, pady=(14, 4))

        self.helper_text = tk.Label(
            self.scores_card,
            text="Each bar shows model confidence by class.",
            font=("Segoe UI", 10),
        )
        self.helper_text.pack(anchor="w", padx=16, pady=(0, 10))

        for class_name in self.class_names:
            row = tk.Frame(self.scores_card)
            row.pack(fill=tk.X, padx=16, pady=6)

            class_label = tk.Label(
                row,
                text=class_name.title(),
                font=("Segoe UI", 10, "bold"),
                width=10,
                anchor="w",
            )
            class_label.pack(side=tk.LEFT)

            value_label = tk.Label(
                row,
                text="0.0%",
                font=("Consolas", 10),
                width=7,
                anchor="e",
            )
            value_label.pack(side=tk.RIGHT)

            score_bar = ttk.Progressbar(
                row,
                orient="horizontal",
                mode="determinate",
                maximum=100,
                style="Score.Horizontal.TProgressbar",
            )
            score_bar.pack(fill=tk.X, padx=(8, 10), expand=True)

            self.class_widgets[class_name] = (row, class_label, value_label, score_bar)

    def _build_history_card(self, parent: tk.Frame) -> None:
        self.history_card = tk.Frame(parent, highlightthickness=1)
        self.history_card.pack(fill=tk.X, pady=(12, 0))

        self.history_title = tk.Label(
            self.history_card,
            text="Recent Predictions",
            font=("Segoe UI", 12, "bold"),
        )
        self.history_title.pack(anchor="w", padx=16, pady=(12, 6))

        self.history_list = tk.Listbox(
            self.history_card,
            height=6,
            activestyle="none",
            borderwidth=0,
            highlightthickness=0,
            font=("Consolas", 9),
        )
        self.history_list.pack(fill=tk.BOTH, padx=16, pady=(0, 12), expand=True)
        self.history_list.insert(tk.END, "No predictions yet")

    def _apply_layout_mode(self, compact: bool) -> None:
        self.compact_mode = compact

        if compact:
            self.body.grid_columnconfigure(0, weight=1)
            self.body.grid_columnconfigure(1, weight=0)
            self.left_panel.grid_configure(row=0, column=0, sticky="nsew", padx=(0, 0), pady=(0, 12))
            self.right_panel.grid_configure(row=1, column=0, sticky="nsew", padx=(0, 0), pady=(0, 0))
            self.preview_frame.config(width=320, height=250)
        else:
            self.body.grid_columnconfigure(0, weight=3)
            self.body.grid_columnconfigure(1, weight=2)
            self.left_panel.grid_configure(row=0, column=0, sticky="nsew", padx=(0, 12), pady=(0, 0))
            self.right_panel.grid_configure(row=0, column=1, sticky="nsew", padx=(0, 0), pady=(0, 0))
            self.preview_frame.config(width=400, height=340)

    def _on_window_resize(self, event: tk.Event) -> None:
        if event.widget != self.root:
            return

        should_be_compact = event.width < 1040
        if should_be_compact != self.compact_mode:
            self._apply_layout_mode(should_be_compact)

    def _apply_theme(self, theme_name: str) -> None:
        self.theme_name = theme_name
        self.root.configure(bg=self.theme["bg"])

        self._set_progress_style()

        self.header.config(bg=self.theme["header"])
        self.title_label.config(bg=self.theme["header"], fg=self.theme["header_text"])
        self.subtitle_label.config(bg=self.theme["header"], fg=self.theme["header_sub"])

        self.theme_label.master.config(bg=self.theme["header"])
        self.theme_label.config(bg=self.theme["header"], fg=self.theme["header_sub"])

        self.body.config(bg=self.theme["bg"])
        self.left_panel.config(bg=self.theme["bg"])
        self.right_panel.config(bg=self.theme["bg"])

        cards = [self.preview_card, self.result_card, self.scores_card]
        for card in cards:
            card.config(bg=self.theme["surface"], highlightbackground=self.theme["card_border"])

        self.history_card.config(bg=self.theme["surface"], highlightbackground=self.theme["card_border"])

        self.preview_title.config(bg=self.theme["surface"], fg=self.theme["text"])
        self.preview_frame.config(bg=self.theme["preview_bg"], highlightbackground=self.theme["preview_border"])
        self.image_label.config(bg=self.theme["preview_bg"], fg=self.theme["muted"])
        self.drop_hint.config(bg=self.theme["surface"], fg=self.theme["muted"])
        self.path_label.config(bg=self.theme["surface"], fg=self.theme["muted"])

        self.upload_button.config(
            bg=self.theme["accent"],
            fg="#ffffff",
            activebackground=self.theme["accent_dark"],
            activeforeground="#ffffff",
        )
        self.predict_button.config(
            bg=self.theme["button_secondary"],
            fg="#ffffff",
            activebackground=self.theme["button_secondary_dark"],
            activeforeground="#ffffff",
        )

        self.result_label.config(bg=self.theme["surface"], fg=self.theme["text"])
        self.confidence_label.config(bg=self.theme["surface"], fg=self.theme["muted"])
        self.top3_title.config(bg=self.theme["surface"], fg=self.theme["text"])
        self.top3_frame.config(bg=self.theme["surface"])
        self.status_label.config(bg=self.theme["surface"])

        self.scores_title.config(bg=self.theme["surface"], fg=self.theme["text"])
        self.helper_text.config(bg=self.theme["surface"], fg=self.theme["muted"])

        self.history_title.config(bg=self.theme["surface"], fg=self.theme["text"])
        self.history_list.config(
            bg=self.theme["bg"],
            fg=self.theme["text"],
            selectbackground=self.theme["accent"],
            selectforeground="#ffffff",
        )

        for _class_name, (row, class_label, value_label, _score_bar) in self.class_widgets.items():
            row.config(bg=self.theme["surface"])
            class_label.config(bg=self.theme["surface"], fg=self.theme["text"])
            value_label.config(bg=self.theme["surface"], fg=self.theme["muted"])

        for chip in self.top3_labels:
            chip.config(bg=self.theme["chip_bg"], fg=self.theme["chip_text"])

        self._set_confidence_meter(self.current_confidence)
        self.status_label.config(fg=self.current_status_color)
        self._bind_hover(self.upload_button, "accent")
        self._bind_hover(self.predict_button, "button_secondary")

    def _on_theme_change(self, selected_theme: str) -> None:
        if selected_theme not in THEMES:
            return
        self._apply_theme(selected_theme)

    def _bind_hover(self, button: tk.Button, color_key: str) -> None:
        normal = self.theme[color_key]
        hover = self.theme["accent_dark"] if color_key == "accent" else self.theme["button_secondary_dark"]

        button.unbind("<Enter>")
        button.unbind("<Leave>")
        button.bind("<Enter>", lambda _event: button.config(bg=hover))
        button.bind("<Leave>", lambda _event: button.config(bg=normal))

    def _configure_drag_and_drop(self) -> None:
        if TKDND_AVAILABLE:
            drop_targets = [
                self.root,
                self.header,
                self.body,
                self.left_panel,
                self.right_panel,
                self.preview_card,
                self.preview_frame,
                self.image_label,
                self.result_card,
                self.scores_card,
                self.history_card,
            ]

            for target in drop_targets:
                target.drop_target_register(DND_FILES)
                target.dnd_bind("<<Drop>>", self._on_drop_file)

            self.drop_hint.config(text="Drag and drop image is enabled.")
        else:
            self.drop_hint.config(
                text="Drag-and-drop unavailable. Install tkinterdnd2 for this feature.",
            )

    def _on_drop_file(self, event: tk.Event) -> None:
        dropped_path = self._extract_drop_path(str(event.data))
        if not dropped_path:
            messagebox.showwarning("Drop failed", "No valid file was dropped.")
            return

        ext = os.path.splitext(dropped_path)[1].lower()
        if ext not in SUPPORTED_IMAGE_EXTENSIONS:
            messagebox.showwarning("Unsupported file", "Please drop a valid image file.")
            return

        self._set_selected_image(dropped_path)

    def _extract_drop_path(self, raw_data: str) -> str | None:
        candidates = self.root.tk.splitlist(raw_data)
        if not candidates:
            return None

        first = candidates[0].strip()
        if first.startswith("{") and first.endswith("}"):
            first = first[1:-1]

        if os.path.isfile(first):
            return first
        return None

    def upload_image(self) -> None:
        image_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")],
        )

        if not image_path:
            return

        self._set_selected_image(image_path)

    def _set_selected_image(self, image_path: str) -> None:
        self.selected_image_path = image_path
        self.show_preview(image_path)
        self.predict_button.config(state=tk.NORMAL)
        self.result_label.config(text="Prediction: -")
        self.confidence_label.config(text="Confidence: -")
        self.current_status_color = self.theme["muted"]
        self.status_label.config(text="Status: Image loaded, ready to predict", fg=self.current_status_color)
        self.path_label.config(text=f"Image path: {os.path.basename(image_path)}")
        self.current_confidence = 0.0
        self.current_meter_color = self.theme["accent"]
        self._set_confidence_meter(0.0)
        self._reset_scores()
        self._update_top3([])

    def show_preview(self, image_path: str) -> None:
        img = Image.open(image_path).convert("RGB")
        img.thumbnail((380, 320))
        self.image_preview = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.image_preview, text="", bg=self.theme["preview_bg"])

    def preprocess_image(self, image_path: str) -> np.ndarray:
        img = image.load_img(image_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array / 255.0

    def _predict_with_tta(self, image_path: str) -> np.ndarray:
        # Average multiple augmented views to reduce single-view misclassification.
        base = Image.open(image_path).convert("RGB")
        variants: list[Image.Image] = [
            base,
            ImageOps.mirror(base),
            base.rotate(-10, resample=Image.Resampling.BILINEAR, fillcolor=(255, 255, 255)),
            base.rotate(10, resample=Image.Resampling.BILINEAR, fillcolor=(255, 255, 255)),
            ImageEnhance.Brightness(base).enhance(0.9),
            ImageEnhance.Brightness(base).enhance(1.1),
            ImageEnhance.Contrast(base).enhance(1.1),
        ]

        batch: list[np.ndarray] = []
        for variant in variants:
            resized = variant.resize(IMG_SIZE, Image.Resampling.BILINEAR)
            batch.append(image.img_to_array(resized) / 255.0)

        stacked = np.stack(batch, axis=0)
        predictions = self.model.predict(stacked, verbose=0)
        return np.mean(predictions, axis=0)

    def predict_image(self) -> None:
        if not self.selected_image_path:
            messagebox.showwarning("No image", "Please upload an image first.")
            return

        try:
            probabilities = self._predict_with_tta(self.selected_image_path)

            predicted_index = int(np.argmax(probabilities))
            predicted_class = self.class_names[predicted_index]
            confidence = float(probabilities[predicted_index] * 100)
            confidence_tone, confidence_color = self._get_confidence_tone(confidence)
            self.current_confidence = confidence
            self.current_meter_color = confidence_color
            self.current_status_color = confidence_color

            self.result_label.config(text=f"Prediction: {predicted_class.title()}")
            self.confidence_label.config(text=f"Confidence: {confidence:.2f}% ({confidence_tone})", fg=confidence_color)
            self.status_label.config(text=f"Status: Prediction completed with {confidence_tone.lower()}", fg=confidence_color)

            self._set_confidence_meter(confidence)
            self._animate_scores(probabilities, predicted_class)
            self._update_top3(probabilities)
            self._append_history(predicted_class, confidence)

        except Exception as exc:
            self.current_status_color = self.theme["warn"]
            self.status_label.config(text="Status: Prediction failed", fg=self.current_status_color)
            messagebox.showerror("Prediction error", str(exc))

    def _get_confidence_tone(self, confidence: float) -> tuple[str, str]:
        if confidence >= 85:
            return "High Confidence", self.theme["confidence_high"]
        if confidence >= 60:
            return "Medium Confidence", self.theme["confidence_mid"]
        return "Low Confidence", self.theme["confidence_low"]

    def _append_history(self, predicted_class: str, confidence: float) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        file_name = os.path.basename(self.selected_image_path or "-")
        record = f"[{stamp}] {predicted_class.title():<10} {confidence:6.2f}%  {file_name}"
        self.history_entries.insert(0, record)
        self.history_entries = self.history_entries[: self.max_history]

        self.history_list.delete(0, tk.END)
        for item in self.history_entries:
            self.history_list.insert(tk.END, item)

    def _set_confidence_meter(self, percent: float) -> None:
        percent = max(0.0, min(100.0, percent))
        meter_width = int(self.confidence_meter.cget("width"))
        meter_height = int(self.confidence_meter.cget("height"))
        fill_width = int((percent / 100.0) * meter_width)

        self.confidence_meter.delete("all")
        self.confidence_meter.create_rectangle(
            0,
            0,
            meter_width,
            meter_height,
            fill=self.theme["meter_trough"],
            outline=self.theme["meter_trough"],
        )
        self.confidence_meter.create_rectangle(
            0,
            0,
            fill_width,
            meter_height,
            fill=self.current_meter_color,
            outline=self.current_meter_color,
        )
        self.confidence_meter.create_text(
            meter_width / 2,
            meter_height / 2,
            text=f"{percent:.1f}%",
            fill="#ffffff" if percent > 28 else self.theme["text"],
            font=("Segoe UI", 9, "bold"),
        )

    def _animate_scores(self, probabilities: np.ndarray, predicted_class: str) -> None:
        for idx, class_name in enumerate(self.class_names):
            score = float(probabilities[idx] * 100)
            self.root.after(idx * 70, lambda name=class_name, value=score: self._update_score_row(name, value))
        self._highlight_prediction(predicted_class)

    def _update_score_row(self, class_name: str, value: float) -> None:
        _row, class_label, value_label, score_bar = self.class_widgets[class_name]
        score_bar["value"] = value
        value_label.config(text=f"{value:4.1f}%")
        class_label.update_idletasks()

    def _highlight_prediction(self, predicted_class: str) -> None:
        for class_name, (_row, class_label, _value_label, _score_bar) in self.class_widgets.items():
            if class_name == predicted_class:
                class_label.config(fg=self.theme["accent_dark"])
            else:
                class_label.config(fg=self.theme["text"])

    def _reset_scores(self) -> None:
        for class_name in self.class_names:
            self._update_score_row(class_name, 0.0)
        self._highlight_prediction("")

    def _update_top3(self, probabilities: list[float] | np.ndarray) -> None:
        if len(probabilities) == 0:
            for idx, chip in enumerate(self.top3_labels, start=1):
                chip.config(text=f"#{idx} -")
            return

        top3_indices = np.argsort(probabilities)[::-1][:3]
        for rank, class_index in enumerate(top3_indices, start=1):
            class_name = self.class_names[int(class_index)].title()
            score = float(probabilities[int(class_index)] * 100)
            self.top3_labels[rank - 1].config(text=f"#{rank} {class_name}: {score:.1f}%")

        if self.top3_labels:
            self.top3_labels[0].config(bg=self.theme["accent"], fg="#ffffff")
            for chip in self.top3_labels[1:]:
                chip.config(bg=self.theme["chip_bg"], fg=self.theme["chip_text"])


def main() -> None:
    if TKDND_AVAILABLE:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()

    app = FruitClassifierGUI(root)
    if app.root.winfo_exists():
        root.mainloop()


if __name__ == "__main__":
    main()
