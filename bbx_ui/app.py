"""BBX Desktop UI Application"""
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, scrolledtext
import subprocess
import os
import yaml
from pathlib import Path

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class BBXStudio(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("🚀 BBX Workflow Studio")
        self.geometry("1200x800")

        # Create main layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(10, weight=1)

        # Logo
        self.logo_label = ctk.CTkLabel(
            self.sidebar,
            text="BBX Studio",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Buttons
        self.new_btn = ctk.CTkButton(
            self.sidebar,
            text="📄 New Workflow",
            command=self.new_workflow
        )
        self.new_btn.grid(row=1, column=0, padx=20, pady=10)

        self.open_btn = ctk.CTkButton(
            self.sidebar,
            text="📁 Open Workflow",
            command=self.open_workflow
        )
        self.open_btn.grid(row=2, column=0, padx=20, pady=10)

        self.save_btn = ctk.CTkButton(
            self.sidebar,
            text="💾 Save Workflow",
            command=self.save_workflow
        )
        self.save_btn.grid(row=3, column=0, padx=20, pady=10)

        self.run_btn = ctk.CTkButton(
            self.sidebar,
            text="▶️ Run Workflow",
            command=self.run_workflow,
            fg_color="green"
        )
        self.run_btn.grid(row=4, column=0, padx=20, pady=10)

        self.validate_btn = ctk.CTkButton(
            self.sidebar,
            text="✓ Validate",
            command=self.validate_workflow
        )
        self.validate_btn.grid(row=5, column=0, padx=20, pady=10)

        # Status
        self.status_label = ctk.CTkLabel(
            self.sidebar,
            text="Ready",
            text_color="gray"
        )
        self.status_label.grid(row=11, column=0, padx=20, pady=10)

        # Main editor area
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # File path label
        self.file_label = ctk.CTkLabel(
            self.main_frame,
            text="No file opened",
            font=ctk.CTkFont(size=14)
        )
        self.file_label.grid(row=0, column=0, sticky="w", padx=10, pady=5)

        # Editor tabs
        self.tabview = ctk.CTkTabview(self.main_frame)
        self.tabview.grid(row=1, column=0, sticky="nsew")

        # Editor tab
        self.tabview.add("Editor")
        self.editor = scrolledtext.ScrolledText(
            self.tabview.tab("Editor"),
            wrap=tk.WORD,
            bg="#2b2b2b",
            fg="#ffffff",
            insertbackground="white",
            font=("Consolas", 11)
        )
        self.editor.pack(fill="both", expand=True)
        self._setup_text_widget_bindings(self.editor)

        # Output tab
        self.tabview.add("Output")
        self.output = scrolledtext.ScrolledText(
            self.tabview.tab("Output"),
            wrap=tk.WORD,
            bg="#1e1e1e",
            fg="#00ff00",
            font=("Consolas", 10)
        )
        self.output.pack(fill="both", expand=True)
        self._setup_text_widget_bindings(self.output)

        self.current_file = None

    def _setup_text_widget_bindings(self, widget):
        """Setup keyboard shortcuts for text widget (Copy, Paste, Select All)"""
        # Copy: Ctrl+C
        widget.bind("<Control-c>", lambda e: self._copy_text(widget))
        # Paste: Ctrl+V
        widget.bind("<Control-v>", lambda e: self._paste_text(widget))
        # Select All: Ctrl+A
        widget.bind("<Control-a>", lambda e: self._select_all(widget))
        # Right-click menu
        widget.bind("<Button-3>", lambda e: self._show_context_menu(widget, e))

    def _copy_text(self, widget):
        """Copy selected text to clipboard"""
        try:
            selected = widget.get(tk.SEL_FIRST, tk.SEL_LAST)
            self.clipboard_clear()
            self.clipboard_append(selected)
        except tk.TclError:
            pass  # No selection
        return "break"

    def _paste_text(self, widget):
        """Paste text from clipboard"""
        try:
            text = self.clipboard_get()
            widget.insert(tk.INSERT, text)
        except tk.TclError:
            pass
        return "break"

    def _select_all(self, widget):
        """Select all text"""
        widget.tag_add(tk.SEL, "1.0", tk.END)
        widget.mark_set(tk.INSERT, "1.0")
        widget.see(tk.INSERT)
        return "break"

    def _show_context_menu(self, widget, event):
        """Show right-click context menu"""
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="Copy (Ctrl+C)", command=lambda: self._copy_text(widget))
        menu.add_command(label="Paste (Ctrl+V)", command=lambda: self._paste_text(widget))
        menu.add_separator()
        menu.add_command(label="Select All (Ctrl+A)", command=lambda: self._select_all(widget))
        menu.tk_popup(event.x_root, event.y_root)

    def new_workflow(self):
        template = """workflow:
  id: my_workflow
  name: My Workflow
  version: "6.0"
  description: A new workflow

  steps:
    - id: step1
      mcp: bbx.logger
      method: info
      inputs:
        message: "Hello from BBX!"
"""
        self.editor.delete("1.0", tk.END)
        self.editor.insert("1.0", template)
        self.file_label.configure(text="Untitled.bbx")
        self.update_status("New workflow created", "blue")

    def open_workflow(self):
        filepath = filedialog.askopenfilename(
            title="Open BBX Workflow",
            filetypes=[("BBX Files", "*.bbx"), ("All Files", "*.*")]
        )
        if filepath:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            self.editor.delete("1.0", tk.END)
            self.editor.insert("1.0", content)
            self.current_file = filepath
            self.file_label.configure(text=os.path.basename(filepath))
            self.update_status(f"Opened {os.path.basename(filepath)}", "blue")

    def save_workflow(self):
        if self.current_file:
            filepath = self.current_file
        else:
            filepath = filedialog.asksaveasfilename(
                defaultextension=".bbx",
                filetypes=[("BBX Files", "*.bbx"), ("All Files", "*.*")]
            )

        if filepath:
            content = self.editor.get("1.0", tk.END)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            self.current_file = filepath
            self.file_label.configure(text=os.path.basename(filepath))
            self.update_status("Saved!", "green")

    def validate_workflow(self):
        # Save first
        if not self.current_file:
            self.save_workflow()

        if self.current_file:
            self.update_status("Validating...", "yellow")
            result = subprocess.run(
                ["python", "cli.py", "validate", self.current_file],
                capture_output=True,
                text=True
            )

            self.log_output(result.stdout + result.stderr)

            if result.returncode == 0:
                self.update_status("✓ Validation passed!", "green")
            else:
                self.update_status("✗ Validation failed", "red")

            self.tabview.set("Output")

    def run_workflow(self):
        if not self.current_file:
            self.save_workflow()

        if self.current_file:
            self.update_status("Running workflow...", "yellow")
            self.log_output(f"\n{'='*60}\n▶️  Running: {self.current_file}\n{'='*60}\n")

            process = subprocess.Popen(
                ["python", "cli.py", "run", self.current_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            # Stream output
            for line in iter(process.stdout.readline, ''):
                if line:
                    self.log_output(line)
                    self.update_idletasks()

            process.wait()

            if process.returncode == 0:
                self.update_status("✓ Completed successfully!", "green")
            else:
                self.update_status("✗ Execution failed", "red")

            self.tabview.set("Output")

    def log_output(self, text):
        self.output.configure(state="normal")
        self.output.insert(tk.END, text)
        self.output.see(tk.END)
        self.output.configure(state="disabled")

    def update_status(self, text, color="gray"):
        self.status_label.configure(text=text, text_color=color)

if __name__ == "__main__":
    app = BBXStudio()
    app.mainloop()
