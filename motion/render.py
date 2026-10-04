# 달빛 아래 경복궁 회랑의 거문고 — motion graphic renderer
import numpy as np, cv2, wave, sys, os, subprocess, math
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1080, 1920, 30
DUR = 188.0
U = '/root/.claude/uploads/74395594-450c-545e-9ddb-8735542722d9/'
SR_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_R = '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc'
FONT_B = '/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc'
GOLD = np.array([120, 200, 255], np.float32) / 255  # BGR warm gold

# ---------------------------------------------------------------- images
def load(name, y0=0, y1=None):
    im = cv2.imread(U + name, cv2.IMREAD_COLOR)
    im = im[y0:y1]
    im = cv2.resize(im, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
    return im

IMG = {}
def init_images():
    IMG['B'] = load('73eb7d4c-image.png', 330)      # bamboo (title cropped off)
    IMG['I'] = load('c2fbc01b-image.png', 300)      # glacier (title cropped off)
    IMG['M'] = load('7bd08557-image.jpg')           # moon
    IMG['D'] = load('3118597f-image.png')           # gold dust
    IMG['P'] = load('d72619d9-image.png')           # palace, man
    IMG['G'] = load('5bd62c66-image.png', 270)      # palace, woman (title cropped off)

def view(key, zoom, cx, cy, shx=0, shy=0, clamp=True):
    im = IMG[key]; h, w = im.shape[:2]
    s = max(W / w, H / h) * zoom
    cx, cy = cx * w, cy * h
    if clamp:
        hw, hh = W / (2 * s), H / (2 * s)
        cx = min(max(cx, hw), w - hw) if w > 2 * hw else w / 2
        cy = min(max(cy, hh), h - hh) if h > 2 * hh else h / 2
    M = np.float32([[s, 0, W / 2 - s * cx + shx], [0, s, H / 2 - s * cy + shy]])
    return cv2.warpAffine(im, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)

# ---------------------------------------------------------------- audio features
def audio_features():
    w = wave.open(os.path.join(SR_DIR, 'a.wav')); sr = w.getframerate()
    x = np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, 2).astype(np.float32).mean(1) / 32768
    n = int(DUR * FPS); hop = sr / FPS; win = 4096
    f = np.fft.rfftfreq(win, 1 / sr)
    bands = [(30, 160), (160, 400), (400, 1200), (1200, 3000), (3000, 7000), (7000, 11000)]
    E = np.zeros((n, len(bands) + 1), np.float32)
    for i in range(n):
        c = int(i * hop); s = x[max(0, c - win // 2):c + win // 2]
        if len(s) < win: s = np.pad(s, (0, win - len(s)))
        F = np.abs(np.fft.rfft(s * np.hanning(win)))
        for j, (lo, hi) in enumerate(bands):
            E[i, j] = np.sqrt((F[(f >= lo) & (f < hi)] ** 2).mean())
        E[i, -1] = np.sqrt((s ** 2).mean())
    E /= np.percentile(E, 97, axis=0) + 1e-9
    E = np.clip(E, 0, 1.5)
    low = E[:, 0]
    slow = np.convolve(low, np.ones(15) / 15, 'same')
    hit = np.clip((low - slow) * 2.2, 0, 1)
    kick = np.zeros(n, np.float32); v = 0
    for i in range(n):
        v = max(hit[i], v * 0.80); kick[i] = v
    # smooth band energies for strings
    sm = np.zeros_like(E); v = E[0].copy()
    for i in range(n):
        v = np.maximum(E[i], v * 0.86); sm[i] = v
    return kick, sm

# ---------------------------------------------------------------- timeline
# shots: (t0, t1, key, z0, z1, (cx0, cy0), (cx1, cy1), particle_mode)
SHOTS = [
    (0.0, 15.2, 'INTRO', 0, 0, (0, 0), (0, 0), 'dust'),
    (15.2, 22.6, 'G', 1.22, 1.0, (0.42, 0.22), (0.5, 0.45), 'petal'),
    (22.6, 37.5, 'P', 1.30, 1.12, (0.66, 0.78), (0.55, 0.55), 'petal'),
    (37.5, 52.3, 'G', 1.45, 1.20, (0.60, 0.72), (0.40, 0.30), 'petal'),
    (52.3, 58.8, 'P', 1.00, 1.10, (0.5, 0.5), (0.5, 0.45), 'petal'),
    (58.8, 65.2, 'G', 1.08, 1.20, (0.30, 0.45), (0.35, 0.40), 'petal'),
    (65.2, 71.6, 'P', 1.35, 1.18, (0.85, 0.25), (0.75, 0.45), 'petal'),
    (71.6, 78.0, 'G', 1.15, 1.35, (0.80, 0.55), (0.75, 0.45), 'petal'),
    (78.0, 81.6, 'M', 0.62, 0.78, (0.5, 0.5), (0.5, 0.5), 'dust'),
    (81.6, 96.5, 'B', 1.40, 1.15, (0.30, 0.40), (0.38, 0.50), 'rain'),
    (96.5, 111.6, 'B', 1.35, 1.10, (0.78, 0.55), (0.70, 0.78), 'rain'),
    (111.6, 123.5, 'I', 1.35, 1.02, (0.50, 0.80), (0.50, 0.15), 'snow'),
    (123.5, 130.8, 'P', 1.05, 1.16, (0.5, 0.5), (0.55, 0.55), 'petal'),
    (130.8, 138.1, 'I', 1.00, 1.12, (0.5, 0.45), (0.45, 0.35), 'snow'),
    (138.1, 145.4, 'G', 1.15, 1.00, (0.5, 0.5), (0.5, 0.5), 'petal'),
    (145.4, 153.0, 'B', 1.00, 1.14, (0.55, 0.5), (0.65, 0.55), 'rain'),
    (153.0, 156.6, 'M', 0.55, 0.72, (0.5, 0.5), (0.5, 0.5), 'dust'),
    (156.6, 164.0, 'G', 1.22, 1.05, (0.35, 0.45), (0.4, 0.4), 'petal'),
    (164.0, 170.6, 'P', 1.55, 1.30, (0.66, 0.74), (0.62, 0.66), 'petal'),
    (170.6, 177.3, 'I', 1.08, 1.25, (0.5, 0.45), (0.5, 0.38), 'snow'),
    (177.3, 184.0, 'P', 1.18, 1.00, (0.5, 0.5), (0.5, 0.5), 'petal'),
    (184.0, 188.1, 'OUTRO', 0, 0, (0, 0), (0, 0), 'dust'),
]
CHORUS = [(52.3, 78.0), (123.5, 153.0), (156.6, 184.0)]
FLASHES = [7.5, 52.3, 81.6, 111.6, 123.5, 156.6, 170.5]

V1 = ['기와지붕을 스치는|서늘한 바람 한 줄기', '수백 년 세월을 품은|주춧돌 위로 울리는 현',
      '묵직한 거문고 술대가|허공을 가를 때', '잠들어 있던 궁궐의 역사가|눈을 뜨네']
CH = ['천둥처럼 울려 퍼져라|묵직한 대북의 박동이여', '달빛을 삼킨 회랑 끝까지|뻗어나가는 기백',
      '서슬 퍼런 선율 속에|깃든 우리의 얼', '달밤의 경복궁을 뒤흔드는|영혼의 외침']
V2 = ['단청 아래로 드리운|깊은 그림자', '외세의 칼바람에도|꺾이지 않던 굳은 절개',
      '술대를 쥔 손끝에|핏줄이 곤두서고', '대지 깊숙이 파고드는|거문고의 깊은 울림']

def block(times, lines, label):
    return [(times[i], times[i + 1], lines[i], label) for i in range(len(lines))]

LYRICS = (block([24.5, 31.3, 38.3, 45.3, 52.0], V1, 'VERSE  I') +
          block([52.3, 58.8, 65.2, 71.6, 77.8], CH, 'CHORUS') +
          block([82.4, 88.8, 95.2, 101.6, 108.0], V2, 'VERSE  II') +
          [(113.0, 123.3, '어둠을 뚫고 솟구치는|백두의 기상이어라', 'BRIDGE')] +
          block([125.0, 132.3, 139.6, 146.9, 154.3], CH, 'CHORUS') +
          block([158.5, 165.4, 172.1, 178.8, 185.3], CH, 'CHORUS'))

# ---------------------------------------------------------------- text sprites
def text_sprite(text, size, font=FONT_B, spacing=0, color=(245, 238, 225), glow=(255, 200, 120), glow_r=14, glow_a=0.9):
    f = ImageFont.truetype(font, size, index=1)
    if spacing:
        widths = [f.getlength(c) + spacing for c in text]; tw = int(sum(widths) - spacing)
    else:
        tw = int(f.getlength(text))
    pad = glow_r * 3
    asc, desc = f.getmetrics()
    im = Image.new('L', (tw + pad * 2, asc + desc + pad * 2), 0)
    d = ImageDraw.Draw(im)
    if spacing:
        x = pad
        for c, wd in zip(text, widths):
            d.text((x, pad), c, font=f, fill=255); x += wd
    else:
        d.text((pad, pad), text, font=f, fill=255)
    a = np.asarray(im, np.float32) / 255
    g = cv2.GaussianBlur(a, (0, 0), glow_r) * glow_a
    g2 = cv2.GaussianBlur(a, (0, 0), glow_r * 0.3) * 0.5
    col = np.array(color[::-1], np.float32) / 255
    gcol = np.array(glow[::-1], np.float32) / 255
    sh = np.clip(cv2.GaussianBlur(a, (0, 0), glow_r * 0.8) * 2.2, 0, 1) * 0.75
    return {'a': a, 'col': col, 'glow': (g + g2)[..., None] * gcol, 'sh': sh[..., None], 'w': a.shape[1], 'h': a.shape[0], 'pad': pad}

def blit(frame, sp, cx, cy, alpha, wipe=1.0, ramp=160):
    h, w = sp['h'], sp['w']
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    fx0, fy0 = max(x0, 0), max(y0, 0); fx1, fy1 = min(x0 + w, W), min(y0 + h, H)
    if fx1 <= fx0 or fy1 <= fy0 or alpha <= 0: return
    sx0, sy0 = fx0 - x0, fy0 - y0; sx1, sy1 = sx0 + fx1 - fx0, sy0 + fy1 - fy0
    m = alpha
    if wipe < 1.0:
        xs = np.arange(sx0, sx1, dtype=np.float32)
        m = alpha * np.clip((wipe * (w + ramp) - xs) / ramp, 0, 1)[None, :, None]
    a = sp['a'][sy0:sy1, sx0:sx1, None] * m
    region = frame[fy0:fy1, fx0:fx1]
    region *= 1 - sp['sh'][sy0:sy1, sx0:sx1] * m
    region += sp['glow'][sy0:sy1, sx0:sx1] * m
    region *= (1 - a)
    region += sp['col'] * a

LY_SP = []; LABEL_SP = {}; TITLE = {}
def init_text():
    for (t0, t1, txt, lab) in LYRICS:
        big = lab == 'BRIDGE'
        rows = [text_sprite(r, 74 if big else 62) for r in txt.split('|')]
        LY_SP.append(rows)
        if lab not in LABEL_SP:
            LABEL_SP[lab] = text_sprite('—   ' + lab + '   —', 26, FONT_R, spacing=6, color=(230, 200, 140), glow_r=8, glow_a=0.5)
    TITLE['l1'] = text_sprite('달빛 아래', 96, FONT_B, spacing=18, glow_r=22)
    TITLE['l2'] = text_sprite('경복궁 회랑의 거문고', 78, FONT_B, spacing=10, glow_r=20)
    TITLE['sub'] = text_sprite('MOONLIT GEOMUNGO  ·  NEO-GUGAK', 26, FONT_R, spacing=8, color=(230, 200, 140), glow_r=8, glow_a=0.5)
    TITLE['cr1'] = text_sprite('제 작', 34, FONT_R, spacing=10, color=(230, 200, 140), glow_r=8, glow_a=0.5)
    TITLE['cr2'] = text_sprite('Free', 92, FONT_B, spacing=12, glow_r=20)
    TITLE['small'] = text_sprite('달빛 아래 경복궁 회랑의 거문고', 30, FONT_R, spacing=4, color=(235, 225, 205), glow_r=8, glow_a=0.4)

# ---------------------------------------------------------------- particles
rng = np.random.default_rng(7)
NP = 140
PART = {
    'x': rng.uniform(0, W, NP), 'y': rng.uniform(0, H, NP), 'sp': rng.uniform(0.4, 1.0, NP),
    'ph': rng.uniform(0, 2 * np.pi, NP), 'sz': rng.uniform(0.5, 1.0, NP), 'rot': rng.uniform(0, 360, NP),
}

def particles(frame, mode, t, k, amt=1.0):
    if amt <= 0: return
    layer = np.zeros((H, W, 3), np.float32)
    P = PART
    if mode == 'petal':
        mask = np.zeros((H, W), np.float32)
        for i in range(90):
            y = (P['y'][i] + t * 90 * P['sp'][i]) % (H + 100) - 50
            x = (P['x'][i] + t * 40 * P['sp'][i] + 60 * math.sin(t * 0.8 + P['ph'][i])) % (W + 100) - 50
            r = 14 * P['sz'][i]; ang = P['rot'][i] + t * 90 * P['sp'][i]
            sq = abs(math.sin(t * 1.7 + P['ph'][i])) * 0.7 + 0.3
            cv2.ellipse(mask, (int(x), int(y)), (int(r), max(1, int(r * 0.55 * sq))), ang, 0, 360, 0.85 * P['sz'][i], -1, cv2.LINE_AA)
        mask = cv2.GaussianBlur(mask, (0, 0), 1.2)[..., None] * amt
        col = np.array([200, 190, 250], np.float32) / 255
        frame *= (1 - mask); frame += col * mask
    if mode in ('petal', 'dust'):
        for i in range(NP):
            y = (P['y'][i] - t * 35 * P['sp'][i]) % H
            x = (P['x'][i] + 25 * math.sin(t * 0.5 + P['ph'][i])) % W
            tw = 0.5 + 0.5 * math.sin(t * 3 + P['ph'][i] * 5)
            b = (0.25 + 0.75 * tw) * (0.6 + 0.8 * k)
            cv2.circle(layer, (int(x), int(y)), 1 + int(2 * P['sz'][i]), (0.35 * b, 0.75 * b, 1.0 * b), -1, cv2.LINE_AA)
    elif mode == 'rain':
        for i in range(NP):
            y = (P['y'][i] + t * 900 * P['sp'][i]) % (H + 200) - 100
            x = (P['x'][i] - y * 0.12) % W
            L = 30 + 50 * P['sp'][i]
            c = 0.30 * P['sz'][i]
            cv2.line(layer, (int(x), int(y)), (int(x + L * 0.12), int(y + L)), (c, c * 0.95, c * 0.85), 1, cv2.LINE_AA)
        for i in range(25):  # bokeh droplets
            x = P['x'][i]; y = (P['y'][i] + t * 12) % H
            b = 0.10 + 0.10 * math.sin(t + P['ph'][i])
            cv2.circle(layer, (int(x), int(y)), int(10 + 25 * P['sz'][i]), (b, b, b * 0.9), -1, cv2.LINE_AA)
    elif mode == 'snow':
        for i in range(NP):
            y = (P['y'][i] + t * 70 * P['sp'][i]) % H
            x = (P['x'][i] + 50 * math.sin(t * 0.6 + P['ph'][i]) - t * 15) % W
            b = 0.55 * P['sz'][i]
            cv2.circle(layer, (int(x), int(y)), 1 + int(3 * P['sz'][i]), (b, b, b), -1, cv2.LINE_AA)
    layer = cv2.GaussianBlur(layer, (0, 0), 1.0)
    frame += layer * amt

# ---------------------------------------------------------------- geomungo strings
def strings(frame, t, en, amt):
    if amt <= 0: return
    layer = np.zeros((H, W, 3), np.float32)
    xs = np.linspace(-20, W + 20, 120)
    for j in range(6):
        y0 = 1660 + j * 34 - 0.03 * (xs - W / 2)    # slight slant like an instrument lying on the floor
        e = float(en[j])
        amp = 2 + 26 * e * (1 - j * 0.08)
        k = j % 3 + 1
        ys = y0 + amp * np.sin(np.pi * (xs + 20) / (W + 40) * k) * math.sin(t * (18 + j * 5) + j)
        pts = np.stack([xs, ys], 1).astype(np.int32)
        c = (0.25 + 0.75 * min(e, 1)) * 0.75
        cv2.polylines(layer, [pts], False, (0.45 * c, 0.80 * c, 1.0 * c), 2, cv2.LINE_AA)
    glow = cv2.GaussianBlur(layer, (0, 0), 6)
    frame += (layer + glow * 1.6) * amt

# ---------------------------------------------------------------- precomputed overlays
YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)
VIG = (1 - 0.65 * np.clip(((XX - W / 2) / (W * 0.75)) ** 2 + ((YY - H * 0.48) / (H * 0.62)) ** 2, 0, 1))[..., None]
BOT = np.clip((YY - 1050) / 600, 0, 1)[..., None] ** 1.1   # darken lower area for lyric legibility
GRAIN = [rng.normal(0, 0.022, (H // 2, W // 2)).astype(np.float32) for _ in range(6)]

def shot_at(t):
    for s in SHOTS:
        if s[0] <= t < s[1]: return s
    return SHOTS[-1]

def render_shot(s, t, k, chorus):
    t0, t1, key, z0, z1, c0, c1, _ = s
    u = ease((t - t0) / (t1 - t0))
    if key == 'INTRO' or key == 'OUTRO':
        lt = t - t0
        if key == 'INTRO':
            bg = view('D', 1.0 + 0.10 * (t / 15.2), 0.5, 0.55 - 0.05 * (t / 15.2)).astype(np.float32) / 255
            my = 0.80 - 0.47 * ease(t / 14.0); mz = 0.40
        else:
            bg = view('D', 1.12 - 0.04 * lt / 4, 0.5, 0.5).astype(np.float32) / 255
            my = 0.36; mz = 0.38 + 0.02 * lt
        moon = view('M', mz, 0.5, 0.5 - (my - 0.5) / mz, clamp=False).astype(np.float32) / 255
        moon = np.clip((moon - 0.04) * 1.1, 0, 1) * 0.92
        return 1 - (1 - bg) * (1 - moon)  # screen blend
    z = z0 + (z1 - z0) * u
    cx = c0[0] + (c1[0] - c0[0]) * u; cy = c0[1] + (c1[1] - c0[1]) * u
    z *= 1 + (0.035 if chorus else 0.015) * k
    shx = shy = 0
    if chorus and k > 0.5:
        r = np.random.default_rng(int(t * 1000))
        shx, shy = r.uniform(-6, 6, 2) * (k - 0.5)
    return view(key, z, cx, cy, shx, shy, clamp=key != 'M').astype(np.float32) / 255

def render(fi, kick, en):
    t = fi / FPS
    k = float(kick[fi]); e = en[fi]
    chorus = any(a <= t < b for a, b in CHORUS)
    s = shot_at(t)
    frame = render_shot(s, t, k, chorus)
    # crossfade into next shot
    XF = 0.7
    idx = SHOTS.index(s)
    if idx + 1 < len(SHOTS) and t > s[1] - XF:
        n = SHOTS[idx + 1]
        a = ease((t - (s[1] - XF)) / XF)
        frame = frame * (1 - a) + render_shot(n, t, k, chorus) * a
    # grade: cool shadows, warm highlights
    lum = frame.mean(2, keepdims=True)
    frame = frame * (0.92 + 0.10 * k) + (1 - lum) * np.array([0.035, 0.012, 0.0], np.float32)
    # bloom
    small = cv2.resize(frame, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    br = np.clip(small - 0.55, 0, 1)
    br = cv2.GaussianBlur(br, (0, 0), 10)
    frame += cv2.resize(br, (W, H), interpolation=cv2.INTER_LINEAR) * (0.9 + 0.8 * k)
    # particles
    particles(frame, s[7], t, k)
    frame *= VIG
    frame *= 1 - 0.55 * BOT
    # strings (absent in intro/outro title sections)
    st_amt = ease((t - 21.0) / 2.0) * (1 - ease((t - 184.0) / 1.5)) * (1.0 if chorus else 0.65)
    strings(frame, t, e, st_amt)
    # flashes
    for ft in FLASHES:
        if ft - 0.05 <= t < ft + 0.9:
            f = math.exp(-max(t - ft, 0) * 5.0)
            frame += np.array([0.75, 0.88, 1.0], np.float32) * 0.55 * f
    # --- text
    if t < 15.6:
        a = ease((t - 2.0) / 2.5) * (1 - ease((t - 13.4) / 2.0))
        rise = 20 * (1 - ease((t - 2.0) / 3.0))
        blit(frame, TITLE['l1'], W / 2, 1180 + rise, a, wipe=ease((t - 2.0) / 3.0), ramp=300)
        blit(frame, TITLE['l2'], W / 2, 1310 + rise, a, wipe=ease((t - 3.2) / 3.0), ramp=300)
        blit(frame, TITLE['sub'], W / 2, 1430, a * ease((t - 5.0) / 2.0))
        ca = a * ease((t - 6.0) / 1.5)
        blit(frame, TITLE['cr1'], W / 2, 1540, ca)
        blit(frame, TITLE['cr2'], W / 2, 1625, ca, wipe=ease((t - 6.2) / 1.4), ramp=200)
    elif t >= 185.4:
        a = ease((t - 185.4) / 0.6) * (1 - ease((t - 187.4) / 0.6))
        blit(frame, TITLE['l1'], W / 2, 1180, a)
        blit(frame, TITLE['l2'], W / 2, 1310, a)
    if t >= 184.3:  # closing production credit
        ca = ease((t - 184.3) / 0.8)
        blit(frame, TITLE['cr1'], W / 2, 1540, ca)
        blit(frame, TITLE['cr2'], W / 2, 1625, ca, wipe=ease((t - 184.4) / 1.0), ramp=200)
    if 15.6 <= t < 185.4:
        sa = ease((t - 16.0) / 2.0) * (1 - ease((t - 184.4) / 1.0))
        blit(frame, TITLE['small'], W / 2, 150, 0.75 * sa)
    if 78.2 <= t < 81.6:  # production credit over the moon interlude
        a = ease((t - 78.2) / 0.6) * (1 - ease((t - 80.9) / 0.6))
        blit(frame, TITLE['cr1'], W / 2, 1500, a)
        blit(frame, TITLE['cr2'], W / 2, 1590 - 10 * ease((t - 78.2) / 3.0), a, wipe=ease((t - 78.4) / 1.2), ramp=200)
    for (t0, t1, txt, lab), rows in zip(LYRICS, LY_SP):
        if t0 - 0.1 <= t < t1 + 0.1:
            fin = ease((t - t0) / 0.5); fout = 1 - ease((t - (t1 - 0.55)) / 0.55)
            a = fin * fout
            lift = -18 * ease((t - (t1 - 0.55)) / 0.55)
            big = lab == 'BRIDGE'
            gap = 100 if big else 86
            base = 1390 - (len(rows) - 1) * gap / 2
            blit(frame, LABEL_SP[lab], W / 2, base - 110 + lift, a * 0.85)
            for r, sp in enumerate(rows):
                wipe = ease((t - t0 - r * 0.9) / (1.3 if not big else 2.5))
                blit(frame, sp, W / 2, base + r * gap + lift, a, wipe=wipe, ramp=220)
    # grain + fades
    g = cv2.resize(GRAIN[fi % 6], (W, H), interpolation=cv2.INTER_NEAREST)[..., None]
    frame += g
    fade = ease(t / 1.2) * (1 - ease((t - 187.2) / 0.8))
    frame *= fade
    return (np.clip(frame, 0, 1) * 255).astype(np.uint8)

def main():
    mode = sys.argv[1]
    init_images(); init_text()
    feats = os.path.join(SR_DIR, 'feats.npz')
    if not os.path.exists(feats):
        kick, en = audio_features(); np.savez(feats, kick=kick, en=en)
    d = np.load(feats); kick, en = d['kick'], d['en']
    if mode == 'still':
        for ts in sys.argv[2:]:
            fi = int(float(ts) * FPS)
            cv2.imwrite(os.path.join(SR_DIR, f'still_{ts}.jpg'), render(fi, kick, en), [cv2.IMWRITE_JPEG_QUALITY, 85])
        return
    a, b, out = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(FPS),
                          '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p', out], stdin=subprocess.PIPE)
    for fi in range(a, b):
        p.stdin.write(render(fi, kick, en).tobytes())
        if fi % 300 == 0: print(out, fi, flush=True)
    p.stdin.close(); p.wait()

if __name__ == '__main__':
    main()
