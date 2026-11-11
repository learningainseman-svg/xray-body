"""Soft-tissue layers (organs + blood vessels) -> organs-data.js

Run through build_skeleton.py, which executes this file when given a third argument:
    python build_skeleton.py <bp3d dir> skeleton-data.js organs-data.js
Every vertex is bound to the one or two nearest bone groups so the tissue bends with the skeleton.
Needs scipy in addition to numpy and fast-simplification.
"""
import collections
from scipy.spatial import cKDTree

out2 = sys.argv[3]
MAIN = ['head', 'neck', 'thorax', 'lumbar', 'pelvis'] + [f'{p}_{s}' for s in 'LR' for p in ('hum', 'fore', 'palm', 'fem', 'shin', 'foot')]


def mainseg(seg):
    if seg.startswith('f_') or seg.startswith('mc1_'): return 'palm_' + seg.split('_')[1]
    if seg.startswith('clav_') or seg.startswith('scap_'): return 'thorax'
    return seg


pts = {m: [] for m in MAIN}
for b in bones.values(): pts[mainseg(b['seg'])].append(b['raw'])
trees = [cKDTree(np.concatenate(pts[m])) for m in MAIN]
ADJ = {('head', 'neck'), ('neck', 'thorax'), ('thorax', 'lumbar'), ('lumbar', 'pelvis')}
for s in 'LR':
    ADJ |= {('thorax', 'hum_' + s), ('hum_' + s, 'fore_' + s), ('fore_' + s, 'palm_' + s),
            ('pelvis', 'fem_' + s), ('fem_' + s, 'shin_' + s), ('shin_' + s, 'foot_' + s)}
adj = np.zeros((len(MAIN), len(MAIN)), bool)
for a, b in ADJ: adj[MAIN.index(a), MAIN.index(b)] = adj[MAIN.index(b), MAIN.index(a)] = True

concept = {}
for l in open(f'{src}/isa_element_parts.txt', encoding='utf8'):
    a = l.rstrip('\n').split('\t'); concept.setdefault(a[2], set()).add(a[0])
in_lung = set()                    # vessels inside the lungs belong to the pulmonary circuit
try:
    for l in open(f'{src}/partof_element_parts.txt', encoding='utf8'):
        a = l.rstrip('\n').split('\t')
        if a[1] in ('left lung', 'right lung'): in_lung.add(a[2])
except FileNotFoundError:
    pass
C_ART, C_VEIN, C_PART, C_PVEIN, C_COR, C_CVEIN, C_PORTAL = 'FMA50720', 'FMA50723', 'FMA66326', 'FMA66643', 'FMA49893', 'FMA12846', 'FMA66645'

# id -> (r, g, b, kind)   kind: 0 organ, 1 flows away from the heart, 2 flows toward the heart, 3 heart
PAL = dict(brain=(.96, .72, .76, 0), cerebellum=(.92, .58, .66, 0), brainstem=(.88, .78, .62, 0), eye=(.88, .94, 1, 0),
           trachea=(.60, .86, .96, 0), lungs=(.52, .80, .96, 0), heart=(.92, .22, .24, 3), esophagus=(.86, .60, .50, 0),
           stomach=(.96, .64, .56, 0), liver=(.66, .30, .22, 0), gallbladder=(.42, .76, .34, 0), pancreas=(.96, .82, .44, 0),
           spleen=(.62, .30, .52, 0), kidney=(.78, .40, .30, 0), adrenal=(.96, .76, .30, 0), ureter=(.92, .86, .50, 0),
           smallint=(.96, .70, .62, 0), largeint=(.80, .54, .44, 0), bladder=(.96, .86, .40, 0))
RED, BLUE = (.96, .20, .18), (.26, .46, .98)
for k in ('artery', 'aorta', 'carotid', 'subclavian', 'brachial', 'radial', 'ulnar', 'renal', 'iliac', 'femoralA', 'popliteal', 'tibial', 'coronary'):
    PAL[k] = RED + (1,)
for k in ('vein', 'venacava', 'jugular', 'femoralV', 'saphenous', 'armveins', 'portal'):
    PAL[k] = BLUE + (2,)
PAL['pulmArt'] = BLUE + (1,)      # pulmonary arteries carry oxygen-poor blood, so they are drawn blue
PAL['pulmVein'] = RED + (2,)
SIDS = [''] + list(PAL)
SID = {n: i for i, n in enumerate(SIDS)}
SIMPLE = {'esophagus': 'esophagus', 'stomach': 'stomach', 'gallbladder': 'gallbladder', 'pancreas': 'pancreas', 'spleen': 'spleen',
          'left kidney': 'kidney', 'right kidney': 'kidney', 'left adrenal gland': 'adrenal', 'right adrenal gland': 'adrenal',
          'left ureter': 'ureter', 'right ureter': 'ureter', 'urinary bladder': 'bladder', 'duodenum': 'smallint',
          'ascending colon': 'largeint', 'transverse colon': 'largeint', 'descending colon': 'largeint', 'rectum': 'largeint',
          'appendix': 'largeint', 'caudate lobe of liver': 'liver'}
VESSEL_IDS = [('aorta', 'aorta'), ('vena cava', 'venacava'), ('carotid artery', 'carotid'), ('jugular vein', 'jugular'),
              ('subclavian artery', 'subclavian'), ('brachial artery', 'brachial'), ('radial artery', 'radial'), ('ulnar artery', 'ulnar'),
              ('renal artery', 'renal'), ('iliac artery', 'iliac'), ('femoral artery', 'femoralA'), ('femoral vein', 'femoralV'),
              ('saphenous vein', 'saphenous'), ('popliteal artery', 'popliteal'), ('tibial artery', 'tibial'),
              ('cephalic vein', 'armveins'), ('basilic vein', 'armveins')]


def soft_class(fid, name):
    """-> (group, id, force_head) or None"""
    n = name.lower(); cs = concept.get(fid, set())
    if 'hepatovenous segment' in n: return 'organs', 'liver', False
    if C_ART in cs or C_VEIN in cs or re.search(r'artery|arterial|vein|aorta|vena cava|pulmonary trunk|celiac trunk|cervical trunk|palmar arch|plantar arch|venous|coronary sinus', n):
        if re.search(r'tendinous|ligament', n): return None
        vein = C_VEIN in cs or re.search(r'vein|vena cava|venous|sinus', n) is not None
        if C_PART in cs or 'pulmonary trunk' in n or 'pulmonary artery' in n or (fid in in_lung and not vein): i = 'pulmArt'
        elif C_PVEIN in cs or 'pulmonary vein' in n or (fid in in_lung and vein): i = 'pulmVein'
        elif C_COR in cs or C_CVEIN in cs or 'coronary' in n: i = 'vein' if vein else 'coronary'
        elif C_PORTAL in cs: i = 'portal'
        else: i = next((v for k, v in VESSEL_IDS if k in n), 'vein' if vein else 'artery')
        return 'vessels', i, False
    if n in ('wall of ventricle', 'wall of left atrium', 'wall of right atrium'): return 'heart', 'heart', False
    if re.search(r'gyrus|occipital lobe|parietal lobule|insula$|white matter of', n): return 'organs', 'brain', True
    if n == 'cerebellum': return 'organs', 'cerebellum', True
    if n in ('pons', 'medulla oblongata', 'midbrain'): return 'organs', 'brainstem', True
    if n.endswith(' sclera') or n.endswith(' cornea'): return 'organs', 'eye', True
    if n == 'trachea': return 'organs', 'trachea', False
    if 'bronch' in n: return 'organs', 'lungs', False
    if n in SIMPLE: return 'organs', SIMPLE[n], False
    if 'jejunum' in n or 'ileum' in n or 'ileocecal' in n: return 'organs', 'smallint', False
    return None


soft, seen = [], set()
for nm in sorted(z.namelist()):
    if not nm.endswith('.obj'): continue
    fid = nm.split('/')[-1][:-4]
    if fid in files: continue
    with z.open(nm) as f: head = f.read(900).decode('utf8', 'replace')
    name = re.search(r'English name : (.*)', head).group(1).strip()
    c = soft_class(fid, name)
    if not c: continue
    _, V, F = load(fid)
    if len(F) == 0: continue
    key = (name, len(V), round(float(V[:, 1].min()), 4))       # the archive repeats some meshes
    if key in seen: continue
    seen.add(key)
    soft.append(dict(name=name, group=c[0], id=c[1], head=c[2], V=V, F=F))

TARGET = dict(organs=85000, heart=9000, vessels=135000)
for grp in TARGET:
    ms = [m for m in soft if m['group'] == grp]
    total = sum(len(m['V']) for m in ms)
    r = min(.94, max(0, 1 - TARGET[grp] / total))
    print(grp, len(ms), 'meshes', total, 'verts -> reduction', round(r, 3))
    for m in ms:
        if r > .02 and len(m['V']) > 60:
            try:
                v, f = fast_simplification.simplify(m['V'].astype(np.float32), m['F'].astype(np.int32), target_reduction=r)
                if len(f) >= 4: m['V'], m['F'] = v.astype(np.float64), f.astype(np.int64)
            except Exception as e:
                print('  simplify failed', m['name'], e)
        m['N'] = normals(m['V'], m['F'])
        V = m['V']; idx = np.arange(len(V))
        if m['head']:
            s1 = np.zeros(len(V), int); s2 = s1.copy(); w = np.zeros(len(V))
        else:
            D = np.stack([t.query(V, workers=-1)[0] for t in trees], 1)
            s1 = D.argmin(1); d1 = D[idx, s1]
            D2 = np.where(adj[s1], D, np.inf); s2 = D2.argmin(1); d2 = D2[idx, s2]
            t = np.clip((d1 / (d1 + d2 + 1e-9) - .25) / .25, 0, 1)
            w = .5 * t * t * (3 - 2 * t)                         # blend only close to the joint between two bone groups
        m['s1'], m['s2'], m['w'] = s1, s2, w

allV = np.concatenate([m['V'] for m in soft]); lo, hi = allV.min(0), allV.max(0)
qc2 = (lo + hi) / 2; qs2 = float((hi - lo).max() / 2 * 1.001)
REC = [('p', '<i2', 3), ('n', 'i1', 3), ('id', 'u1'), ('s', 'u1', 2), ('w', 'u1'), ('pad', 'u1', 3)]     # 16 bytes
vbuf, ibuf, chunks = [], [], []
st = dict(v0=0, i0=0, base=0, icount=0)


def flush(grp):
    if st['base']: chunks.append([grp, st['v0'], st['base'], st['i0'], st['icount']])
    st['v0'] += st['base']; st['i0'] += st['icount']; st['base'] = st['icount'] = 0


for grp in ('organs', 'heart', 'vessels'):
    for m in soft:
        if m['group'] != grp: continue
        n = len(m['V'])
        if st['base'] + n > 65535: flush(grp)
        rec = np.zeros(n, dtype=REC)
        rec['p'] = np.round((m['V'] - qc2) / qs2 * 32767).astype(np.int16)
        rec['n'] = np.round(m['N'] * 127).astype(np.int8)
        rec['id'] = SID[m['id']]; rec['s'][:, 0] = m['s1']; rec['s'][:, 1] = m['s2']; rec['w'] = np.round(m['w'] * 255)
        vbuf.append(rec.tobytes()); ibuf.append((m['F'] + st['base']).astype('<u2').tobytes())
        st['base'] += n; st['icount'] += m['F'].size
    flush(grp)

SLOW = {}
for m in soft: SLOW.setdefault(m['name'].lower(), m)
SLAB = [('organs', 'brain', 'Left middle frontal gyrus', 1), ('organs', 'cerebellum', 'Cerebellum', 2), ('organs', 'brainstem', 'Pons', 2),
        ('organs', 'eye', 'Left sclera', 2), ('organs', 'trachea', 'Trachea', 1), ('organs', 'lungs', 'Right main bronchus proper', 1),
        ('heart', 'heart', 'Wall of ventricle', 1), ('organs', 'esophagus', 'Esophagus', 2), ('organs', 'stomach', 'Stomach', 1),
        ('organs', 'liver', 'Hepatovenous segment VIII', 1), ('organs', 'gallbladder', 'Gallbladder', 2), ('organs', 'pancreas', 'Pancreas', 2),
        ('organs', 'spleen', 'Spleen', 1), ('organs', 'kidney', 'Left kidney', 1), ('organs', 'kidney', 'Right kidney', 1),
        ('organs', 'adrenal', 'Left adrenal gland', 2), ('organs', 'ureter', 'Right ureter', 2), ('organs', 'smallint', 'Middle part of ileum', 1),
        ('organs', 'largeint', 'Descending colon', 1), ('organs', 'largeint', 'Ascending colon', 2), ('organs', 'bladder', 'Urinary bladder', 1),
        ('vessels', 'aorta', 'Arch of aorta', 1), ('vessels', 'aorta', 'Abdominal aorta', 2), ('vessels', 'venacava', 'Superior vena cava', 1),
        ('vessels', 'venacava', 'Inferior vena cava', 2), ('vessels', 'pulmArt', 'Pulmonary trunk', 1), ('vessels', 'pulmVein', 'Left superior pulmonary vein', 2),
        ('vessels', 'coronary', 'Trunk of left coronary artery', 2), ('vessels', 'portal', 'Hepatic portal vein', 2)]
for S in ('Left', 'Right'):
    SLAB += [('vessels', 'carotid', f'{S} common carotid artery', 1), ('vessels', 'jugular', f'{S} internal jugular vein', 1),
             ('vessels', 'subclavian', f'{S} subclavian artery', 2), ('vessels', 'brachial', f'{S} brachial artery', 1),
             ('vessels', 'radial', f'{S} radial artery', 1), ('vessels', 'ulnar', f'{S} ulnar artery', 2),
             ('vessels', 'armveins', f'{S} cephalic vein', 2), ('vessels', 'renal', f'{S} renal artery', 2),
             ('vessels', 'iliac', f'{S} common iliac artery', 2), ('vessels', 'femoralA', f'{S} femoral artery', 1),
             ('vessels', 'femoralV', f'{S} femoral vein', 2), ('vessels', 'saphenous', f'{S} great saphenous vein', 1),
             ('vessels', 'popliteal', f'{S} popliteal artery', 2), ('vessels', 'tibial', f'{S} anterior tibial artery', 1)]
slabels = []
for grp, lid, mn, lvl in SLAB:
    m = SLOW.get(mn.lower())
    if m is None or m['id'] != lid:
        print('  label skipped:', mn, m and m['id']); continue
    V = m['V']; c = V.mean(0)
    k = int(np.argmin(np.linalg.norm(V - c, axis=1) + .6 * (V[:, 2] - V[:, 2].min())))
    slabels.append(dict(id=lid, grp=grp, lvl=lvl, seg=int(m['s1'][k]), p=[round(float(x), 5) for x in V[k]]))
hv = SLOW['wall of ventricle']['V'].mean(0)
vb, ib = b''.join(vbuf), b''.join(ibuf)
meta2 = dict(ids=SIDS, pal=[[0, 0, 0, 0]] + [list(PAL[k]) for k in SIDS[1:]], main=MAIN, chunks=chunks, labels=slabels,
             heart=[round(float(x), 4) for x in hv], qc=[float(x) for x in qc2], qs=qs2, vBytes=len(vb))
with open(out2, 'w', encoding='utf8', newline='\n') as f:
    f.write('/* Anatomy meshes: ' + meta['credit'] + '. Simplified and re-packed; shared under the same licence. */\n')
    f.write('window.SOFT=' + json.dumps(meta2, separators=(',', ':')) + ';\nwindow.SOFT.data="' + base64.b64encode(vb + ib).decode() + '";\n')
print('soft verts', st['v0'], 'tris', st['i0'] // 3, 'bytes', len(vb) + len(ib), 'chunks', [(c[0], c[2]) for c in chunks])
print(collections.Counter(m['id'] for m in soft))
