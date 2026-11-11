"""Builds skeleton-data.js from the BodyParts3D "is_a" OBJ archive.

Source data: BodyParts3D, (c) The Database Center for Life Science,
licensed under CC Attribution-Share Alike 2.1 Japan.
  https://dbarchive.biosciencedbc.jp/data/bodyparts3d/LATEST/

Usage: python build_skeleton.py <dir with isa_BP3D_4.0_obj_99.zip + isa_element_parts.txt> <out.js>
Needs: numpy, fast-simplification
"""
import sys, re, json, zipfile, base64
import numpy as np
import fast_simplification

src, out_path = sys.argv[1], sys.argv[2]
z = zipfile.ZipFile(f'{src}/isa_BP3D_4.0_obj_99.zip')
WANT = {'FMA5018', 'FMA12516', 'FMA7591', 'FMA7486', 'FMA7487', 'FMA7488'}  # bones, teeth, costal cartilage, sternum parts
files = set()
for l in open(f'{src}/isa_element_parts.txt', encoding='utf8'):
    a = l.rstrip('\n').split('\t')
    if a[0] in WANT:
        files.add(a[2])

IDS = ['', 'frontal', 'parietal', 'temporal', 'occipital', 'sphenoid', 'nasal', 'zygomatic', 'maxilla', 'mandible', 'teeth',
       'hyoid', 'cervical', 'thoracic', 'lumbar', 'sacrum', 'clavicle', 'scapula', 'manubrium', 'sternum', 'xiphoid',
       'trueribs', 'falseribs', 'floatribs', 'cartilage', 'humerus', 'radius', 'ulna', 'carpals', 'metacarpals', 'phalanges',
       'hipbone', 'femur', 'patella', 'tibia', 'fibula', 'talus', 'calcaneus', 'tarsals', 'metatarsals', 'toes',
       'cranium', 'ribs']
ID = {n: i for i, n in enumerate(IDS)}
ORD = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth', 'eleventh', 'twelfth']


def classify(name):
    n = name.lower()
    s = 'L' if 'left' in n else 'R' if 'right' in n else ''
    if 'tooth' in n: return 'head', 'teeth', ''
    if 'costal cartilage' in n: return 'thorax', 'cartilage', ''
    if 'manubrium' in n: return 'thorax', 'manubrium', ''
    if 'body of sternum' in n: return 'thorax', 'sternum', ''
    if 'xiphoid' in n: return 'thorax', 'xiphoid', ''
    if n.endswith(' rib'):
        k = next(i for i, o in enumerate(ORD) if f' {o} ' in n)
        return 'thorax', 'trueribs' if k < 7 else 'falseribs' if k < 10 else 'floatribs', 'ribs'
    if 'thoracic vertebra' in n: return 'thorax', 'thoracic', ''
    if 'cervical vertebra' in n or n in ('atlas', 'axis'): return 'neck', 'cervical', ''
    if 'hyoid' in n: return 'neck', 'hyoid', ''
    if 'lumbar vertebra' in n: return 'lumbar', 'lumbar', ''
    if n == 'sacrum': return 'pelvis', 'sacrum', ''
    if 'hip bone' in n: return 'pelvis', 'hipbone', ''
    for k in ('frontal', 'parietal', 'temporal', 'occipital', 'sphenoid'):
        if k in n: return 'head', k, 'cranium'
    if any(k in n for k in ('ethmoid', 'vomer', 'lacrimal', 'palatine', 'concha')): return 'head', '', 'cranium'
    if 'nasal bone' in n: return 'head', 'nasal', ''
    if 'zygomatic' in n: return 'head', 'zygomatic', ''
    if 'maxilla' in n: return 'head', 'maxilla', ''
    if 'mandible' in n: return 'head', 'mandible', ''
    if 'clavicle' in n: return 'clav_' + s, 'clavicle', ''
    if 'scapula' in n: return 'scap_' + s, 'scapula', ''
    if 'humerus' in n: return 'hum_' + s, 'humerus', ''
    if 'radius' in n: return 'fore_' + s, 'radius', ''
    if 'ulna' in n: return 'fore_' + s, 'ulna', ''
    if any(k in n for k in ('scaphoid', 'lunate', 'triquetral', 'pisiform', 'trapezium', 'trapezoid', 'capitate', 'hamate')):
        return 'palm_' + s, 'carpals', ''
    if 'metacarpal' in n: return ('mc1_' if 'first' in n else 'palm_') + s, 'metacarpals', ''
    if 'phalanx' in n and ('finger' in n or 'thumb' in n):
        f = next(k for k in ('thumb', 'index', 'middle', 'ring', 'little') if k in n)
        p = next(k for k in ('proximal', 'middle phalanx', 'distal') if n.startswith(k))[:4].strip()
        return f'f_{s}_{f}_{p}', 'phalanges', ''
    if 'femur' in n: return 'fem_' + s, 'femur', ''
    if 'patella' in n: return 'shin_' + s, 'patella', ''
    if 'tibia' in n: return 'shin_' + s, 'tibia', ''
    if 'fibula' in n: return 'shin_' + s, 'fibula', ''
    if 'talus' in n: return 'foot_' + s, 'talus', ''
    if 'calcaneus' in n: return 'foot_' + s, 'calcaneus', ''
    if any(k in n for k in ('cuboid', 'cuneiform', 'navicular')): return 'foot_' + s, 'tarsals', ''
    if 'metatarsal' in n: return 'foot_' + s, 'metatarsals', ''
    if 'phalanx' in n and 'toe' in n: return 'foot_' + s, 'toes', ''
    if 'sesamoid' in n: return 'foot_' + s, '', ''
    raise ValueError('unclassified: ' + name)


def load(fid):
    t = z.read(f'isa_BP3D_4.0_obj_99/{fid}.obj').decode('utf8', 'replace')
    name = re.search(r'English name : (.*)', t).group(1).strip()
    V, F = [], []
    for l in t.split('\n'):
        if l.startswith('v '):
            V.append([float(x) for x in l.split()[1:4]])
        elif l.startswith('f '):
            idx = [int(p.split('/')[0]) - 1 for p in l.split()[1:]]
            for i in range(1, len(idx) - 1):
                F.append([idx[0], idx[i], idx[i + 1]])
    V = np.array(V, np.float64); F = np.array(F, np.int64)
    # canonical frame (metres): x = subject's left, y = down, z = back  (matches MediaPipe world axes for a person facing the camera)
    V = np.stack([V[:, 0], -V[:, 2], V[:, 1]], 1) / 1000.0
    return name, V, F


def simplify(V, F):
    n = len(V)
    r = 0 if n < 250 else .35 if n < 900 else .55 if n < 3000 else .72
    if r == 0: return V, F
    v, f = fast_simplification.simplify(V.astype(np.float32), F.astype(np.int32), target_reduction=r)
    return v.astype(np.float64), f.astype(np.int64)


def normals(V, F):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    N = np.zeros_like(V)
    for k in range(3): np.add.at(N, F[:, k], fn)
    l = np.linalg.norm(N, axis=1, keepdims=True); l[l == 0] = 1
    return N / l


bones = {}  # name -> dict
for fid in sorted(files):
    name, V, F = load(fid)
    if len(F) == 0: continue
    seg, bid, grp = classify(name)
    key = name if name not in bones else name + ' #2'
    bones[key] = dict(name=name, raw=V, rawF=F, seg=seg, id=ID[bid], grp=ID[grp])
print(len(bones), 'parts')


def B(name): return bones[name]['raw']
def cen(v): return v.mean(0)
def side(s): return 'Left' if s == 'L' else 'Right'


def ends(V, near):
    """centroids of the two ends along the principal axis; first = the end closer to `near`"""
    c = V - V.mean(0)
    w, vec = np.linalg.eigh(c.T @ c)
    p = c @ vec[:, -1]
    lo, hi = np.quantile(p, .15), np.quantile(p, .85)
    a, b = V[p <= lo].mean(0), V[p >= hi].mean(0)
    return (a, b) if np.linalg.norm(a - near) < np.linalg.norm(b - near) else (b, a)


J = {}
for s in 'LR':
    S = side(s); sx = 1 if s == 'L' else -1
    h = B(f'{S} humerus'); y0, y1 = h[:, 1].min(), h[:, 1].max()
    J['shoulder_' + s] = cen(h[h[:, 1] < y0 + .10 * (y1 - y0)])
    J['elbow_' + s] = cen(h[h[:, 1] > y1 - .05 * (y1 - y0)])
    r, u = B(f'{S} radius'), B(f'{S} ulna')
    rd = cen(r[r[:, 1] > r[:, 1].max() - .06 * np.ptp(r[:, 1])]); ud = cen(u[u[:, 1] > u[:, 1].max() - .06 * np.ptp(u[:, 1])])
    J['radD_' + s], J['ulnD_' + s], J['wrist_' + s] = rd, ud, (rd + ud) / 2
    w = J['wrist_' + s]
    J['mc2_' + s] = ends(B(f'{S} second metacarpal bone'), w)[1]
    J['mc5_' + s] = ends(B(f'{S} fifth metacarpal bone'), w)[1]
    a, b = ends(B(f'{S} first metacarpal bone'), w); J[f'mc1_{s}_a'], J[f'mc1_{s}_b'] = a, b
    for f in ('thumb', 'index', 'middle', 'ring', 'little'):
        for p, pn in (('prox', 'Proximal phalanx'), ('midd', 'Middle phalanx'), ('dist', 'Distal phalanx')):
            nm = f'{pn} of {S.lower()} {f}' + ('' if f == 'thumb' else ' finger')
            if nm not in bones: continue
            a, b = ends(B(nm), w); J[f'f_{s}_{f}_{p}_a'], J[f'f_{s}_{f}_{p}_b'] = a, b
    fm = B(f'{S} femur'); y0, y1 = fm[:, 1].min(), fm[:, 1].max()
    top = fm[fm[:, 1] < y0 + .14 * (y1 - y0)]
    med = top[np.abs(top[:, 0]) <= np.quantile(np.abs(top[:, 0]), .45)]
    J['hip_' + s] = cen(med)
    J['knee_' + s] = cen(fm[fm[:, 1] > y1 - .05 * (y1 - y0)])
    J['ankle_' + s] = cen(B(f'{S} talus'))
    c = B(f'{S} calcaneus'); J['heel_' + s] = cen(c[c[:, 2] >= np.quantile(c[:, 2], .85)])
    J['toe_' + s] = cen(B(f'Distal phalanx of {S.lower()} big toe'))
    cl = B(f'{S} clavicle'); ax = np.abs(cl[:, 0])
    J['clavM_' + s] = cen(cl[ax <= np.quantile(ax, .08)]); J['clavL_' + s] = cen(cl[ax >= np.quantile(ax, .92)])
    m = B('Mandible'); m = m[m[:, 0] * sx > 0]
    back = m[m[:, 2] >= np.quantile(m[:, 2], .8)]
    cond = cen(back[back[:, 1] <= np.quantile(back[:, 1], .1)])
    J['ear_' + s] = cond + np.array([sx * .030, -.004, .008])   # out to the skin surface, where the tracker puts the ear
nb = np.concatenate([B('Left nasal bone'), B('Right nasal bone')])
tip = nb[np.argmax(nb[:, 1])]
J['nose'] = np.array([0, tip[1] + .024, tip[2] - .030])
J['c1'] = cen(B('Atlas'))
J['c7t1'] = (cen(B('Seventh cervical vertebra')) + cen(B('First thoracic vertebra'))) / 2
J['t12l1'] = (cen(B('Twelfth thoracic vertebra')) + cen(B('First lumbar vertebra'))) / 2
sa = B('Sacrum'); J['l5s1'] = (cen(B('Fifth lumbar vertebra')) + cen(sa[sa[:, 1] < np.quantile(sa[:, 1], .1)])) / 2
J['coccyx'] = cen(sa[sa[:, 1] > np.quantile(sa[:, 1], .97)])

# simplify + normals
for b in bones.values():
    b['V'], b['F'] = simplify(b['raw'], b['rawF'])
    b['N'] = normals(b['V'], b['F'])

LOW = {k.lower(): k for k in bones}
LABELS = [('cranium', 'Right parietal bone', 1), ('frontal', 'Frontal bone', 2), ('parietal', 'Left parietal bone', 2),
          ('temporal', 'Left temporal bone', 2), ('temporal', 'Right temporal bone', 2), ('occipital', 'Occipital bone', 2),
          ('sphenoid', 'Sphenoid bone', 2), ('nasal', 'Right nasal bone', 2), ('zygomatic', 'Left zygomatic bone', 2),
          ('zygomatic', 'Right zygomatic bone', 2), ('maxilla', 'Right maxilla', 2), ('mandible', 'Mandible', 1),
          ('teeth', 'Right upper central secondary incisor tooth', 2), ('hyoid', 'Hyoid bone', 2),
          ('cervical', 'Fourth cervical vertebra', 1), ('thoracic', 'Ninth thoracic vertebra', 1),
          ('lumbar', 'Third lumbar vertebra', 1), ('sacrum', 'Sacrum', 1), ('manubrium', 'Manubrium', 2),
          ('sternum', 'Body of sternum', 1), ('xiphoid', 'Xiphoid process', 2), ('ribs', 'Right fifth rib', 1),
          ('trueribs', 'Left third rib', 2), ('falseribs', 'Left ninth rib', 2), ('floatribs', 'Right eleventh rib', 2),
          ('cartilage', 'Right sixth costal cartilage', 2)]
for S in ('Left', 'Right'):
    s_ = S.lower()
    LABELS += [('clavicle', f'{S} clavicle', 1), ('scapula', f'{S} scapula', 1), ('humerus', f'{S} humerus', 1),
               ('radius', f'{S} radius', 1), ('ulna', f'{S} ulna', 1), ('carpals', f'{S} capitate', 1),
               ('metacarpals', f'{S} third metacarpal bone', 1), ('phalanges', f'Proximal phalanx of {s_} middle finger', 1),
               ('hipbone', f'{S} hip bone', 1), ('femur', f'{S} femur', 1), ('patella', f'{S} patella', 1),
               ('tibia', f'{S} tibia', 1), ('fibula', f'{S} fibula', 1), ('talus', f'{S} talus', 2),
               ('calcaneus', f'{S} calcaneus', 2), ('tarsals', f'Navicular bone of {s_} foot', 1),
               ('metatarsals', f'{S} third metatarsal bone', 1), ('toes', f'Proximal phalanx of {s_} second toe', 2)]
labels = []
for lid, bn, lvl in LABELS:
    b = bones[LOW[bn.lower()]]; V = b['V']; c = V.mean(0)
    p = V[np.argmin(np.linalg.norm(V - c, axis=1) + .6 * (V[:, 2] - V[:, 2].min()))]
    labels.append(dict(id=lid, lvl=lvl, seg=b['seg'], p=[round(float(x), 5) for x in p]))

# pack
order = sorted({b['seg'] for b in bones.values()})
allV = np.concatenate([b['V'] for b in bones.values()])
lo, hi = allV.min(0), allV.max(0); qc = (lo + hi) / 2; qs = float((hi - lo).max() / 2 * 1.001)
vbuf, ibuf, segs, v0, i0 = [], [], {}, 0, 0
for seg in order:
    vs, base = v0, 0
    icount = 0
    for b in bones.values():
        if b['seg'] != seg: continue
        n = len(b['V'])
        rec = np.zeros(n, dtype=[('p', '<i2', 3), ('n', 'i1', 3), ('id', 'u1'), ('g', 'u1'), ('pad', 'u1')])
        rec['p'] = np.round((b['V'] - qc) / qs * 32767).astype(np.int16)
        rec['n'] = np.round(b['N'] * 127).astype(np.int8)
        rec['id'] = b['id']; rec['g'] = b['grp']
        vbuf.append(rec.tobytes())
        ibuf.append((b['F'] + base).astype('<u2').tobytes())
        base += n; icount += b['F'].size
    assert base < 65536, (seg, base)
    segs[seg] = [vs, base, i0, icount]
    v0 += base; i0 += icount
vb, ib = b''.join(vbuf), b''.join(ibuf)
meta = dict(ids=IDS, segs=segs, joints={k: [round(float(x), 5) for x in v] for k, v in J.items()}, labels=labels,
            qc=[float(x) for x in qc], qs=qs, vBytes=len(vb), verts=v0, indices=i0,
            credit='BodyParts3D, (c) The Database Center for Life Science licensed under CC Attribution-Share Alike 2.1 Japan')
with open(out_path, 'w', encoding='utf8', newline='\n') as f:
    f.write('/* Anatomy meshes: ' + meta['credit'] + '. Simplified and re-packed; shared under the same licence. */\n')
    f.write('window.SKEL=' + json.dumps(meta, separators=(',', ':')) + ';\nwindow.SKEL.data="' + base64.b64encode(vb + ib).decode() + '";\n')
print('verts', v0, 'tris', i0 // 3, 'bytes', len(vb) + len(ib), 'segments', len(segs))
for k in ('shoulder_L', 'elbow_L', 'wrist_L', 'hip_L', 'knee_L', 'ankle_L', 'heel_L', 'toe_L', 'ear_L', 'ear_R', 'nose', 'c1', 'mc2_L', 'mc5_L', 'radD_L', 'ulnD_L'):
    print(k, np.round(J[k], 3))


if len(sys.argv) > 3:      # also build the organ and blood-vessel layers
    exec(open(__file__.replace("build_skeleton.py", "build_soft.py"), encoding="utf8").read())
