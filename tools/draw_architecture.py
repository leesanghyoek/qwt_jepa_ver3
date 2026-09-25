"""Paper-style architecture figure for QWT-JEPA, written as plain SVG.

Usage: python3 tools/draw_architecture.py docs/kien_truc.svg
"""
import sys
from xml.sax.saxutils import escape

W, H = 1280, 1416
FONT = "Segoe UI, Roboto, 'Noto Sans', Helvetica, Arial, sans-serif"
C = {  # fill, stroke
    'data':   ('#F5F5F5', '#616161'),
    'tf':     ('#E8EAF6', '#3949AB'),
    'bb':     ('#E3F2FD', '#1E88E5'),
    'lat':    ('#F3E5F5', '#8E24AA'),
    'p1':     ('#FFF8E1', '#F9A825'),
    'color':  ('#FFF3E0', '#EF6C00'),
    'edge':   ('#E8F5E9', '#2E7D32'),
    'out':    ('#E0F2F1', '#00897B'),
    'loss':   ('#FCE4EC', '#D81B60'),
}
el = []

def text(x, y, s, size=15, weight='normal', color='#1A1A1A', anchor='middle', style=''):
    el.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{color}" '
              f'text-anchor="{anchor}" {style}>{escape(s)}</text>')

def node(x, y, w, h, kind, title, sub=(), rx=10, title_size=16):
    fill, stroke = C[kind]
    el.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    lines = [title] + list(sub)
    total = 20 + 17 * (len(lines) - 1)
    ty = y + h / 2 - total / 2 + 15
    text(x + w / 2, ty, title, size=title_size, weight='bold')
    for i, s in enumerate(sub):
        text(x + w / 2, ty + 20 + 17 * i, s, size=13, color='#37474F')
    return (x, y, w, h)

def region(x, y, w, h, kind, label, dash='7 5', right=False):
    fill, stroke = C[kind]
    el.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{fill}" fill-opacity="0.35" '
              f'stroke="{stroke}" stroke-width="2" stroke-dasharray="{dash}"/>')
    text(x + w - 14 if right else x + 14, y + 22, label, size=15, weight='bold', color=stroke,
         anchor='end' if right else 'start')

def arrow(points, color='#455A64', width=2.2, dash=None, label=None, lx=None, ly=None, lsize=13, lanchor='middle'):
    d = 'M ' + ' L '.join(f'{px} {py}' for px, py in points)
    extra = f' stroke-dasharray="{dash}"' if dash else ''
    marker = 'url(#ah)' if color == '#455A64' else f'url(#ah-{color[1:]})'
    el.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"{extra} marker-end="{marker}"/>')
    if label:
        text(lx, ly, label, size=lsize, color=color, anchor=lanchor, style='font-style="italic"')

def mid_right(b): x, y, w, h = b; return (x + w, y + h / 2)
def mid_left(b):  x, y, w, h = b; return (x, y + h / 2)
def mid_top(b):   x, y, w, h = b; return (x + w / 2, y)
def mid_bot(b):   x, y, w, h = b; return (x + w / 2, y + h)

# ---------------- regions
region(190, 150, 490, 320, 'bb', '① Backbone — train ở phase 1, đóng băng ở phase 2')
region(842, 110, 423, 360, 'edge', '③ Phase 2 — decoder khôi phục', right=True)
region(190, 505, 810, 200, 'p1', '② Phase 1 — học latent (chỉ lúc train)')

# ---------------- inputs
img_in = node(20, 205, 150, 70, 'data', 'Ảnh mờ', ['3 × 256 × 256'])
imu_in = node(20, 365, 150, 70, 'data', 'IMU nhiễu', ['6 × 128'])
# ---------------- backbone
qwt = node(210, 205, 135, 70, 'tf', 'QWT Hilbert', ['48 × 128 × 128'])
haar = node(210, 365, 135, 70, 'tf', 'Haar', ['12 × 64'])
enc_i = node(370, 205, 135, 70, 'bb', 'Encoder ảnh', ['CNN 4 stage'])
enc_u = node(370, 365, 135, 70, 'bb', 'Encoder IMU', ['CNN 4 stage'])
fus = node(535, 280, 125, 80, 'bb', 'Fusion', ['có cổng'])
# ---------------- latent
zi = node(705, 205, 110, 70, 'lat', 'ZI', ['128 × 16 × 16'])
zu = node(705, 365, 110, 70, 'lat', 'ZU', ['128 × 8'])
# ---------------- phase 2: image decoder group
el.append('<rect x="855" y="150" width="200" height="185" rx="10" fill="#FFFFFF" fill-opacity="0.7" stroke="#2E7D32" stroke-width="1.5"/>')
text(955, 170, 'Decoder ảnh · phễu–loa', size=14, weight='bold', color='#2E7D32')
colb = node(870, 180, 170, 62, 'color', 'Nhánh MÀU', ['U-Net 128² · màu + độ sáng'], title_size=15)
edgb = node(870, 262, 170, 62, 'edge', 'Nhánh ĐƯỜNG NÉT', ['U-Net 256² · kênh Y'], title_size=15)
join = node(1080, 222, 58, 58, 'out', 'Ghép', [], rx=29, title_size=14)
img_out = node(1160, 212, 95, 78, 'out', 'Ảnh', ['phục hồi'])
imu_dec = node(870, 365, 170, 70, 'edge', 'Decoder IMU', ['Haar · skip từ encoder'], title_size=15)
imu_out = node(1160, 365, 95, 70, 'out', 'IMU', ['phục hồi'])
# ---------------- phase 1
clean = node(20, 565, 150, 70, 'data', 'Ảnh + IMU', ['SẠCH'])
teach = node(205, 555, 135, 80, 'p1', 'Teacher EMA', ['bản sao 2 encoder', 'đọc bản SẠCH'])
loss1 = node(420, 550, 230, 90, 'loss', 'Loss phase 1', ['JEPA = ½ (ảnh + IMU)', 'VICReg · neo · Jacobian'])
pred_u = node(705, 555, 110, 70, 'p1', 'Predictor', ['IMU'])
pred_i = node(862, 555, 115, 70, 'p1', 'Predictor', ['ảnh'])

# ---------------- arrows: backbone
arrow([mid_right(img_in), mid_left(qwt)]); arrow([mid_right(imu_in), mid_left(haar)])
arrow([mid_right(qwt), mid_left(enc_i)]); arrow([mid_right(haar), mid_left(enc_u)])
arrow([mid_right(enc_i), (520, 240), (520, 305), (535, 305)])
arrow([mid_right(enc_u), (520, 400), (520, 335), (535, 335)])
arrow([(660, 305), (685, 305), (685, 240), (705, 240)])
arrow([(660, 335), (685, 335), (685, 400), (705, 400)])
# ---------------- arrows: phase 2
arrow([mid_right(zi), (855, 240)])
arrow([(95, 205), (95, 72), (955, 72), (955, 150)], color='#2E7D32', width=3,
      label='skip: chính ảnh mờ 256 × 256 → cho biết cạnh nằm ở đâu', lx=520, ly=63)
arrow([(955, 242), (955, 262)], color='#EF6C00', width=1.8)
text(1008, 256, 'độ sáng', size=12, color='#EF6C00', anchor='start', style='font-style="italic"')
arrow([mid_right(colb), (1060, 211), (1060, 240), (1080, 240)])
arrow([mid_right(edgb), (1060, 293), (1060, 262), (1080, 262)])
arrow([mid_right(join), mid_left(img_out)])
arrow([mid_right(zu), mid_left(imu_dec)])
arrow([mid_right(imu_dec), mid_left(imu_out)])
# ---------------- arrows: phase 1
arrow([mid_right(clean), mid_left(teach)])
arrow([mid_right(teach), (420, 595)], label='đích TI, TU', lx=380, ly=585)
arrow([mid_bot(zu), mid_top(pred_u)], label='ZU', lx=772, ly=500, lanchor='start')
el.append('<path d="M 815 262 L 829 262 L 829 393 A 7 7 0 0 1 829 407 L 829 590 L 862 590" '
          'fill="none" stroke="#455A64" stroke-width="2.2" marker-end="url(#ah)"/>')
text(835, 500, 'ZI', size=13, color='#455A64', anchor='start', style='font-style="italic"')
arrow([mid_left(pred_u), (650, 590)], label='đoán TU', lx=678, ly=582)
arrow([mid_bot(pred_i), (919, 672), (535, 672), (535, 640)], label='đoán TI', lx=740, ly=665)
arrow([(490, 555), (490, 435)], color='#F9A825', dash='6 4', width=2)
text(498, 500, 'Jacobian: nhạy với cạnh, điếc với nhiễu', size=12, color='#B26A00', anchor='start', style='font-style="italic"')

# ---------------- inside the image decoder: funnel and loudspeaker (U-Net) CNNs
el.append('<line x1="20" y1="728" x2="1260" y2="728" stroke="#B0BEC5" stroke-width="1.5"/>')
text(20, 760, 'Bên trong decoder ảnh: mỗi nhánh là một CNN phễu–loa (U-Net)', size=19, weight='bold', color='#37474F', anchor='start')
text(20, 784, 'Phễu thu nhỏ ÷2 mỗi tầng xuống đáy 16×16, nơi latent ZI (cũng 16×16) đi vào · loa phóng ×2 trở lại · '
     'skip đưa đặc trưng gần của mỗi tầng từ phễu sang loa', size=13, color='#607D8B', anchor='start')

def unet(x0, y0, levels, channels, kind, title, input_label, output_label, dx=70, dy=80, bw=30):
    """U-Net drawing: block height ~ resolution, funnel left, loudspeaker right."""
    fill, stroke = C[kind]
    L = len(levels)
    heights = [132, 96, 68, 48, 32][5 - L:]
    yc = [y0 + 80 + i * dy for i in range(L)]
    xf = [x0 + i * dx for i in range(L)]                      # funnel; xf[-1] is the bottom
    xl = [x0 + (2 * (L - 1) - i) * dx for i in range(L)]      # loudspeaker
    text(x0 + (L - 1) * dx + bw / 2, y0 + 4, title, size=15, weight='bold', color=stroke)
    def block(x, i, fill=fill, stroke=stroke):
        h = heights[i]
        el.append(f'<rect x="{x}" y="{yc[i] - h / 2}" width="{bw}" height="{h}" rx="4" fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>')
    for i in range(L - 1):
        block(xf[i], i); block(xl[i], i)
        label = f'{levels[i]}² · {channels[i]} kênh'
        text(xf[i] + bw / 2, yc[i] - heights[i] / 2 - 7, label, size=11, color='#37474F')
        text(xl[i] + bw / 2, yc[i] - heights[i] / 2 - 7, label, size=11, color='#37474F')
        el.append(f'<path d="M {xf[i] + bw} {yc[i]} L {xl[i]} {yc[i]}" stroke="#90A4AE" stroke-width="1.6" '
                  f'stroke-dasharray="5 4" fill="none" marker-end="url(#ah-90A4AE)"/>')
        arrow([(xf[i] + bw / 2, yc[i] + heights[i] / 2), (xf[i] + bw / 2, yc[i + 1]), (xf[i + 1], yc[i + 1])], color=stroke, width=1.8)
        arrow([(xl[i + 1] + bw, yc[i + 1]), (xl[i] + bw / 2, yc[i + 1]),
               (xl[i] + bw / 2, yc[i] + heights[i] / 2)], color=stroke, width=1.8)
    text((xf[0] + xl[0] + bw) / 2, yc[0] - 5, 'skip: đặc trưng gần', size=11, color='#78909C',
         style='font-style="italic"')
    xb = xf[-1]                                              # bottom: the latent joins here
    block(xb, L - 1, fill=C['lat'][0], stroke=C['lat'][1])
    arrow([(xb + bw / 2, yc[-1] + 70), (xb + bw / 2, yc[-1] + heights[-1] / 2)], color=C['lat'][1], width=2.2)
    text(xb + bw / 2, yc[-1] + 86, 'ZI 128 × 16 × 16 (JEPA)', size=12, weight='bold', color=C['lat'][1])
    text(xb + bw / 2 - 8, yc[-1] + heights[-1] / 2 + 20, f'đáy {levels[-1]}² · {channels[-1]} kênh', size=11,
         color='#37474F', anchor='end')
    top = yc[0] - heights[0] / 2
    arrow([(xf[0] + bw / 2, top - 46), (xf[0] + bw / 2, top - 22)], color='#455A64', width=1.8)
    text(xf[0] - 4, top - 52, input_label, size=12, weight='bold', color='#37474F', anchor='start')
    arrow([(xl[0] + bw / 2, top - 22), (xl[0] + bw / 2, top - 46)], color='#455A64', width=1.8)
    text(xl[0] + bw + 4, top - 52, output_label, size=12, weight='bold', color='#37474F', anchor='end')
    text(x0 + (L - 1) * dx / 2 + bw / 2 - 20, yc[-1] + 6, 'PHỄU', size=14, weight='bold', color=stroke)
    text(x0 + 3 * (L - 1) * dx / 2 + bw / 2 + 20, yc[-1] + 6, 'LOA', size=14, weight='bold', color=stroke)

unet(70, 850, [256, 128, 64, 32, 16], [16, 24, 32, 48, 56], 'edge', 'Nhánh ĐƯỜNG NÉT (kênh Y) · 0,63 M',
     'vào: Y ảnh mờ + độ sáng nền', 'ra: chi tiết Y (mọi cạnh)')
unet(800, 850, [128, 64, 32, 16], [12, 16, 24, 32], 'color', 'Nhánh MÀU · 0,16 M',
     'vào: ảnh mờ thu nhỏ ½', 'ra: ảnh nền màu', dx=62)

# how a CNN level is built
el.append('<rect x="20" y="1362" width="1240" height="40" rx="8" fill="#F5F7F8" stroke="#CFD8DC" stroke-width="1"/>')
text(36, 1387, 'Mỗi khối CNN = conv 3×3 + khối residual (conv–ReLU–conv + identity)  ·  phễu: conv 4×4 bước 2 (÷2)  ·  '
     'loa: conv + pixel shuffle (×2), nối với skip rồi conv  ·  conv cuối zero-init  ·  bật bằng split_branch_arch: unet',
     size=13, color='#37474F', anchor='start')

markers = ''.join(
    f'<marker id="ah-{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
    f'<path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>' for c in ('#2E7D32', '#EF6C00', '#F9A825', '#90A4AE', '#1E88E5', '#8E24AA'))
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="{FONT}">'
       f'<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
       f'<path d="M0,0 L10,5 L0,10 z" fill="#455A64"/></marker>{markers}</defs>'
       f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>' + ''.join(el) + '</svg>')
open(sys.argv[1], 'w', encoding='utf-8').write(svg)
