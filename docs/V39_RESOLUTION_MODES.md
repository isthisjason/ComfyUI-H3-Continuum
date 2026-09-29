# V3.9 resolution modes

V3.9 exposes **Megapixel + Aspect Ratio** and **Manual Width × Height**.
V3.8 retains its existing Size Source UI, schema, and migration behavior.

In megapixel mode, choose Draft 0.30 MP, Balanced 0.60 MP, Native 768, or
Custom (0.10–16.00 MP). The aspect choices, in order, are:

Auto / First Image, 1:1, 4:3, 3:2, 16:10, 16:9, 21:9, 3:4, 2:3, 10:16, 9:16.

Fixed ratios are independent of First Image. Auto uses First Image's aspect,
not its native pixel dimensions. References from the Fantastic adapter do not
select the output aspect. First Image preprocessing may stretch its contents
when the chosen canvas ratio differs; this extension does not change that
preprocessing or introduce cropping, padding, or interpolation.

The existing sizing math rounds each dimension to the nearest 32 pixels, with
halfway values rounded upward. The requested ratio can therefore change slightly.
**Resolved Size** shows the selected aspect, resolved Width × Height, and actual
MP after rounding. For Auto, unavailable browser image geometry is shown as
pending/unknown; the backend resolves the actual tensor geometry at execution.
Browser geometry is a preview, not a substitute for the backend result.

| Ratio | Draft | Balanced | Native 768 |
| --- | --- | --- | --- |
| 1:1 | 544×544 | 768×768 | 768×768 |
| 4:3 | 640×480 | 896×672 | 1024×768 |
| 3:2 | 672×448 | 960×640 | 1152×768 |
| 16:10 | 704×448 | 992×608 | 1216×768 |
| 16:9 | 736×416 | 1024×576 | 1344×768 |
| 21:9 | 832×352 | 1184×512 | 1344×576 |
| 3:4 | 480×640 | 672×896 | 768×1024 |
| 2:3 | 448×672 | 640×960 | 768×1152 |
| 10:16 | 448×704 | 608×992 | 768×1216 |
| 9:16 | 416×736 | 576×1024 | 768×1344 |

For example, 3:2 Draft is 0.301056 MP and Balanced is 0.614400 MP. Native
768 retains the existing 768 short-edge / 1344 long-edge-cap policy; it is not
a fixed megapixel target.

Manual uses the existing validation unchanged: integer dimensions from 32 to
16384 inclusive, each divisible by 32. This is a structural range, not a GPU
memory guarantee. Invalid values are not silently rounded or clamped.

## Saved workflows and API values

No serialized widget is added or reordered. The V3.9 `aspect` dropdown alone
gains new ratio strings; the old strings `Auto from First Image`, `Square 1:1`,
`Landscape 16:9`, and `Portrait 9:16` are retained behind the shorter UI labels.
The `size_source` dropdown gains `Megapixel + Aspect Ratio`; Manual still stores
`Manual`. New fixed ratios require the new mode in API requests.

Loading and preview refresh do not rewrite any sizing values. Mode switching
preserves inactive aspect, preset, Custom MP, Width, and Height. A deliberate
mode or aspect selection opts into the new mode. Merely opening an old workflow
does not opt it in.

The historical source values remain meaningful:

- `First Image`: Auto regardless of the saved legacy aspect. Missing First Image
  uses stored manual Width/Height and emits the existing warning.
- `Legacy Aspect` (also the default for omitted API values): retains the original
  aspect resolver. Missing Auto image uses the existing 16:9 preset fallback.
- `Manual`: retains exact stored dimensions and existing validation.
- New `Megapixel + Aspect Ratio` with Auto: uses the First Image/manual-fallback
  behavior, not Legacy Aspect's 16:9 fallback.

An old workflow already saved as Manual stays Manual; the UI does not infer a
previous preset from its dimensions. Older plugin versions may not understand
newly saved V3.9 aspect/mode values. Existing resolution-dependent Run Storage
and continuation contracts continue to apply. No workflow JSON, Fantastic,
Second Pass, or frame interpolation changes are included.
