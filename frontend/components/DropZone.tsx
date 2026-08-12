/**
 * OGB — OrbitalGuard
 * DropZone — drag-and-drop / file-picker image upload area.
 */
"use client";

import { useRef, useState, DragEvent, ChangeEvent } from "react";
import { Upload } from "lucide-react";

const ACCEPTED = ["image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"];
const MAX_MB = 20;

interface Props {
  onFile: (file: File) => void;
  disabled?: boolean;
}

export default function DropZone({ onFile, disabled }: Props) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  function validate(file: File): string | null {
    if (!ACCEPTED.includes(file.type))
      return `Unsupported type "${file.type}". Accepted: JPEG, PNG, WebP, BMP, TIFF.`;
    if (file.size > MAX_MB * 1024 * 1024)
      return `File exceeds ${MAX_MB} MB limit (${(file.size / 1024 / 1024).toFixed(1)} MB).`;
    return null;
  }

  function handleFile(file: File) {
    const err = validate(file);
    if (err) {
      alert(`OGB: ${err}`);
      return;
    }
    onFile(file);
  }

  function onDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  }

  function onChange(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
    // reset so same file can be re-selected
    e.target.value = "";
  }

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      onClick={() => !disabled && inputRef.current?.click()}
      className={[
        "flex flex-col items-center justify-center gap-3 w-full rounded border-2 border-dashed",
        "cursor-pointer select-none transition-colors duration-150 py-12",
        dragging
          ? "border-[#3b82f6] bg-[#1a2236]"
          : "border-[#2d3748] bg-[#0f1624] hover:border-[#3b82f6] hover:bg-[#141c2e]",
        disabled ? "opacity-40 pointer-events-none" : "",
      ].join(" ")}
      role="button"
      aria-label="Upload spacecraft camera image"
    >
      <Upload className="text-[#3b82f6]" size={36} strokeWidth={1.5} />
      <p className="text-[#a0aec0] text-sm font-mono">
        Drag & drop an image or{" "}
        <span className="text-[#3b82f6] underline underline-offset-2">browse</span>
      </p>
      <p className="text-[#4a5568] text-xs font-mono">
        JPEG · PNG · WebP · BMP · TIFF — max {MAX_MB} MB
      </p>
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED.join(",")}
        onChange={onChange}
        className="hidden"
        disabled={disabled}
      />
    </div>
  );
}
