"""Layer-by-layer figure of the funnel-and-loudspeaker CNN (UNetBranch).

Shapes, parameter counts and receptive fields are the ones printed from the
model built by qjepa.models.decoders.UNetBranch with the recipe's widths.

Usage: python3 tools/draw_unet_detail.py docs/phe_loa_chi_tiet.svg
"""
import sys
from xml.sax.saxutils import escape

W, H = 1300, 1250
FONT = "Segoe UI, Roboto, 'Noto Sans', Helvetica, Arial, sans-serif"
EDGE = ('#E8F5E9', '#2E7D32')
LAT = ('#F3E5F5', '#8E24AA')
DATA = ('#F5F5F5', '#616161')
OUT = ('#E0F2F1', '#00897B')
el = []


def text(x, y, s, size=14, weight='normal', color='#1A1A1A', anchor='middle', extra=''):
    el.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{color}" '
              f'text-anchor="{anchor}" {extra}>{escape(s)}</text>')


def box(x, y, w, h, colours, lines, sizes=(15, 13), bold_first=True, rx=8):
    fill, stroke = colours
    el.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    total = sizes[0] + (len(lines) - 1) * (sizes[1] + 5)
    ty = y + h / 2 - total / 2 + sizes[0] - 3
    for i, line in enumerate(lines):
        size = sizes[0] if i == 0 else sizes[1]
        text(x + w / 2, ty, line, size=size, weight='bold' if i == 0 and bold_first else 'normal',
             color='#1A1A1A' if i == 0 else '#37474F')
        ty += (sizes[1] + 5) if i == 0 else (sizes[1] + 5)
    return x, y, w, h


def arrow(points, color='#455A64', width=2, dash=None):
    d = 'M ' + ' L '.join(f'{px} {py}' for px, py in points)
    dash = f' stroke-dasharray="{dash}"' if dash else ''
    el.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"{dash} marker-end="url(#m{color[1:]})"/>')


def label(x, y, s, color='#455A64', anchor='start', size=12):
    text(x, y, s, size=size, color=color, anchor=anchor, extra='font-style="italic"')


# ---------------------------------------------------------------- title
text(30, 38, 'Chi tiết CNN phễu–loa (U-Net) — nhánh ĐƯỜNG NÉT', size=22, weight='bold', anchor='start')
text(30, 62, 'kênh sáng Y · ảnh 256 × 256 · 626.169 tham số · shape ghi theo [kênh × cao × rộng]',
     size=14, color='#607D8B', anchor='start')

# ---------------------------------------------------------------- funnel (left) and loudspeaker (right)
levels = [(256, 16, 7), (128, 24, 18), (64, 32, 40), (32, 48, 84)]   # resolution, channels, receptive field px
rows = [275, 405, 525, 635]
BW, BH = 300, 62
lx = [70 + i * 50 for i in range(4)]
rx = [930 - i * 50 for i in range(4)]

# input and stem above the funnel, tail and output above the loudspeaker
box(lx[0], 82, BW, 50, DATA, ['Vào: Y ảnh mờ + độ sáng nền', '[2 × 256 × 256]'])
box(lx[0], 160, BW, 50, EDGE, ['stem: conv 3×3 + ReLU', '→ [16 × 256 × 256]'])
arrow([(lx[0] + 60, 132), (lx[0] + 60, 160)])
arrow([(lx[0] + 60, 210), (lx[0] + 60, rows[0] - BH / 2)])
box(rx[0], 82, BW, 50, OUT, ['Ra: chi tiết Y = chi tiết Y ảnh mờ + Δ', '[1 × 256 × 256] · mọi cạnh'])
box(rx[0], 160, BW, 50, EDGE, ['tail: conv 3×3 (khởi tạo 0)', '→ Δ [1 × 256 × 256]'])
arrow([(rx[0] + BW - 60, 160), (rx[0] + BW - 60, 132)])
arrow([(rx[0] + BW - 60, rows[0] - BH / 2), (rx[0] + BW - 60, 210)])

for i, (res, ch, rf) in enumerate(levels):
    y = rows[i] - BH / 2
    box(lx[i], y, BW, BH, EDGE, [f'Khối residual · tầng {res}²', f'[{ch} × {res} × {res}] · vùng nhìn {rf} px'])
    box(rx[i], y, BW, BH, EDGE, ['conv 3×3 + ReLU → khối residual', f'[{ch} × {res} × {res}]'])
    cx = rx[i] - 34                                            # concat node
    el.append(f'<circle cx="{cx}" cy="{rows[i]}" r="15" fill="#FFFFFF" stroke="#455A64" stroke-width="2"/>')
    text(cx, rows[i] + 5, 'nối', size=11, weight='bold', color='#455A64')
    arrow([(cx + 15, rows[i]), (rx[i], rows[i])])
    # skip: shallow features of this level go straight across
    arrow([(lx[i] + BW, rows[i]), (cx - 15, rows[i])], color='#90A4AE', dash='6 4')
    label((lx[i] + BW + cx) / 2, rows[i] - 8, f'skip · đặc trưng tầng {res}²',
          color='#78909C', anchor='middle')
    if i < 3:
        nres, nch, _ = levels[i + 1]
        # funnel: stride-2 conv down to the next level
        arrow([(lx[i] + 60, rows[i] + BH / 2), (lx[i] + 60, rows[i + 1]), (lx[i + 1], rows[i + 1])], color='#2E7D32')
        label(lx[i] + 68, rows[i] + BH / 2 + 22, f'↓ conv 4×4 bước 2 + ReLU → [{nch} × {nres}²]', color='#2E7D32')
        # loudspeaker: from the level below up into this level's concat node
        arrow([(rx[i + 1] + BW - 60, rows[i + 1] - BH / 2), (rx[i + 1] + BW - 60, rows[i] + 48),
               (cx, rows[i] + 48), (cx, rows[i] + 15)], color='#2E7D32')
        label(cx + 8, rows[i] + 66, f'↑ pixel shuffle ×2 → [{ch} × {res}²]', color='#2E7D32')

# ---------------------------------------------------------------- bottom of the funnel
bx, by, bw, bh = 400, 720, 500, 110
box(bx, by, bw, bh, LAT, ['ĐÁY 16 × 16 · vùng nhìn 204 px ≈ cả ảnh',
                          'nối [56 × 16²] với ZI qua conv 1×1 (128 → 56) → [112 × 16²]',
                          'conv 3×3 + ReLU → khối residual → [56 × 16 × 16]'])
arrow([(lx[3] + 60, rows[3] + BH / 2), (lx[3] + 60, by + bh / 2), (bx, by + bh / 2)], color='#2E7D32')
label(lx[3] + 68, rows[3] + BH / 2 + 22, '↓ conv 4×4 bước 2 + ReLU → [56 × 16²]', color='#2E7D32')
arrow([(bx + bw, by + bh / 2), (rx[3] + BW - 60, by + bh / 2), (rx[3] + BW - 60, rows[3] + BH / 2)], color='#2E7D32')
label(bx + bw + 8, by + bh / 2 + 22, '↑ pixel shuffle ×2 → [48 × 32²]', color='#2E7D32')
box(bx + 110, 870, 280, 50, LAT, ['ZI · latent JEPA', '[128 × 16 × 16] · cùng lưới 16 × 16'])
arrow([(bx + 250, 870), (bx + 250, by + bh)], color='#8E24AA', width=2.5)

# ---------------------------------------------------------------- shallow -> deep bar
el.append('<defs><linearGradient id="depth" x1="0" y1="0" x2="0" y2="1">'
          '<stop offset="0" stop-color="#C8E6C9"/><stop offset="1" stop-color="#6A1B9A"/></linearGradient></defs>')
el.append('<rect x="22" y="245" width="16" height="585" rx="8" fill="url(#depth)"/>')
text(30, 237, 'NÔNG', size=12, weight='bold', color='#2E7D32')
text(30, 848, 'SÂU', size=12, weight='bold', color='#6A1B9A')
text(54, 430, 'cạnh, chi tiết nhỏ', size=11, color='#2E7D32', anchor='start', extra='transform="rotate(-90 54 430)"')
text(54, 820, 'bố cục, vật thể, độ sáng chung', size=11, color='#6A1B9A', anchor='start', extra='transform="rotate(-90 54 820)"')

# ---------------------------------------------------------------- legend: residual block and the three operations
el.append('<rect x="30" y="950" width="1240" height="150" rx="10" fill="#FAFAFA" stroke="#CFD8DC"/>')
text(50, 978, 'Khối residual', size=15, weight='bold', anchor='start')
mini = [('x', 60), ('conv 3×3', 150), ('ReLU', 250), ('conv 3×3', 350), ('+', 450), ('ra', 530)]
for name, x in mini:
    if name in ('x', 'ra'):
        text(x, 1025, name, size=14, weight='bold')
    elif name == '+':
        el.append(f'<circle cx="{x}" cy="1020" r="13" fill="#FFFFFF" stroke="#455A64" stroke-width="2"/>')
        text(x, 1025, '+', size=16, weight='bold')
    else:
        el.append(f'<rect x="{x - 42}" y="1004" width="84" height="32" rx="6" fill="#E8F5E9" stroke="#2E7D32" stroke-width="1.5"/>')
        text(x, 1025, name, size=13)
for a, b in [(72, 108), (192, 222), (278, 308), (392, 437), (463, 518)]:
    arrow([(a, 1020), (b, 1020)], width=1.6)
arrow([(60, 1034), (60, 1062), (450, 1062), (450, 1033)], width=1.6, color='#90A4AE')
label(250, 1080, 'đường tắt: cộng lại đầu vào (identity)', color='#78909C', anchor='middle')

notes = ['↓ PHỄU · conv 4×4 bước 2 + ReLU: ảnh nhỏ đi ½, nhiều kênh hơn, nhìn rộng hơn → đặc trưng sâu dần',
         '↑ LOA · conv 3×3 → pixel shuffle ×2 + ReLU: ảnh lớn gấp đôi, dựng lại chi tiết theo đặc trưng sâu',
         'nối (concat) với skip: đưa lại đặc trưng nông cùng tầng, để loa không mất chi tiết nhỏ',
         'tail khởi tạo 0: lúc đầu Δ = 0, đầu ra đúng bằng chi tiết Y của ảnh mờ — chỉ có thể tốt dần lên']
for i, s in enumerate(notes):
    text(610, 988 + 27 * i, s, size=13, color='#37474F', anchor='start')

# ---------------------------------------------------------------- colour branch
el.append('<rect x="30" y="1120" width="1240" height="110" rx="10" fill="#FFF3E0" stroke="#EF6C00" stroke-width="1.5"/>')
text(50, 1148, 'Nhánh MÀU · cùng cấu trúc, chạy trên ảnh mờ thu nhỏ ½ (128 × 128) · 164.883 tham số',
     size=15, weight='bold', color='#E65100', anchor='start')
text(50, 1176, 'vào [3 × 128²] → stem [12 × 128²] → phễu: 128² · 12 kênh (nhìn 14 px) → 64² · 16 (36 px) → 32² · 24 (80 px) '
     '→ đáy 16² · 32 + ZI (≈ cả ảnh)', size=13, color='#37474F', anchor='start')
text(50, 1202, '→ loa ngược lại 32² → 64² → 128², nối skip mỗi tầng → tail → Δ [3 × 128²] + ảnh mờ ½ = ảnh nền màu '
     '(màu Cb, Cr và độ sáng nền)   · vùng nhìn tính theo px của ảnh 256', size=13, color='#37474F', anchor='start')

colours = ['#455A64', '#2E7D32', '#90A4AE', '#8E24AA']
markers = ''.join(f'<marker id="m{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
                  f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>' for c in colours)
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="{FONT}">'
       f'<defs>{markers}</defs><rect width="{W}" height="{H}" fill="#FFFFFF"/>' + ''.join(el) + '</svg>')
open(sys.argv[1], 'w', encoding='utf-8').write(svg)
