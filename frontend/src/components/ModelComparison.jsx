import { useState } from 'react'
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { Empty, fmtNum } from './Bits.jsx'

const COLORS = { fcn: '#8A8F98', unet: '#2C6E91', deeplabv3plus: '#3E8E6A', segformer: '#E8A42A', swin: '#7B5EA7', ensemble: '#C8322B' }

// what each architecture does differently (used in the "why" explanation)
const ABOUT = {
  fcn: 'Simple baseline: no skip connections, so slick edges come out blurry.',
  unet: 'Skip connections copy fine detail from the encoder to the decoder, so thin and narrow slicks keep sharp edges.',
  deeplabv3plus: 'Atrous (dilated) convolutions look at several scales at once, which helps with large slicks and wide context.',
  segformer: 'Transformer: self-attention sees the whole image, so it separates oil from dark sea patterns using global context. Light (MiT-B1).',
  swin: 'Transformer with shifted-window attention. Strong, but the heaviest and slowest single model here.',
  ensemble: 'Averages the probabilities of U-Net, DeepLabV3+ and SegFormer: their errors partly cancel out, and where they disagree the system shows an uncertainty map.',
}

const COLS = [
  ['accuracy', 'Accuracy'], ['precision', 'Precision'], ['recall', 'Recall'], ['f1', 'F1 (oil)'], ['iou', 'IoU (oil)'], ['auc', 'ROC-AUC'],
]

function Why({ c }) {
  const rows = c.models
  const by = (k) => rows.find((r) => r.key === k)
  const best = by(c.best)
  const single = by(c.best_single)
  const ens = by('ensemble')
  const fcn = by('fcn')
  const cm = rows[0].confusion_matrix
  const oilShare = (cm[1][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
  const pts = (a, b, k) => ((a[k] - b[k]) * 100).toFixed(1)
  return (
    <div className="empty" style={{ marginTop: '1rem' }}>
      <h3>Which model is best, and why</h3>
      <ul style={{ margin: '.5rem 0 0', paddingLeft: '1.2rem', lineHeight: 1.6 }}>
        <li><b>Best overall: {best.label}</b>, with oil F1 {fmtNum(best.f1, 3)}, IoU {fmtNum(best.iou, 3)} and ROC-AUC {fmtNum(best.auc, 3)}.</li>
        {ens && best.key === 'ensemble' && single && (
          <li>The ensemble beats the best single model ({single.label}) by {pts(ens, single, 'f1')} F1 points and {pts(ens, single, 'auc')} AUC points. {ABOUT.ensemble}</li>
        )}
        {ens && best.key !== 'ensemble' && (
          <li>The best single model ({best.label}) is {pts(best, ens, 'f1')} F1 points ahead of the ensemble, a very small gap. The ensemble is still deployed because it is more stable across different images and gives a per-pixel uncertainty map, so analysts can see where the detection is doubtful.</li>
        )}
        {ens && single && <li>Trade-off: the ensemble runs 3 models, so it is slower ({fmtNum(ens.ms_per_image, 0)} ms per image vs {fmtNum(single.ms_per_image, 0)} ms for {single.label}).</li>}
        {fcn && best && <li>Compared with the FCN baseline, {best.label} improves oil F1 by {pts(best, fcn, 'f1')} points and IoU by {pts(best, fcn, 'iou')} points.</li>}
        <li>Why accuracy is not enough: accuracy counts sea and oil pixels together ({(oilShare * 100).toFixed(1)}% of the test pixels are oil). It does not show how well the slick itself is outlined{fcn ? ` (FCN still reaches ${(fcn.accuracy * 100).toFixed(1)}% accuracy, but its oil IoU is only ${fmtNum(fcn.iou, 3)})` : ''}. So the main ranking uses F1 and IoU on the oil class, with ROC-AUC showing how well each model separates oil from sea at every threshold.</li>
      </ul>
      <table className="table" style={{ marginTop: '.8rem' }}>
        <thead><tr><th>Model</th><th>How it works / strength</th></tr></thead>
        <tbody>{rows.map((r) => <tr key={r.key}><td>{r.label}</td><td>{ABOUT[r.key] || ''}</td></tr>)}</tbody>
      </table>
    </div>
  )
}

export default function ModelComparison({ c }) {
  if (!c) {
    return (
      <section className="panel">
        <h2>Detection models: accuracy, ROC curve and AUC</h2>
        <Empty title="Comparison not run yet">Run <code>python -m ml.segmentation.compare_models</code> from the project folder, then reload this page.</Empty>
      </section>
    )
  }
  return <Compare c={c} />
}

const PRESETS = [
  ['all', 'All models', null],
  ['used', 'Models used in the project (ensemble and its 3 members)', ['unet', 'deeplabv3plus', 'segformer', 'ensemble']],
  ['cnn_tf', 'CNN vs Transformer models', ['fcn', 'unet', 'deeplabv3plus', 'segformer', 'swin']],
  ['base_dep', 'Baseline vs deployed (FCN vs Ensemble)', ['fcn', 'ensemble']],
  ['ens_best', 'Ensemble vs best single model', 'ens_best'],
  ['custom', 'Custom: choose models…', 'custom'],
]

function Compare({ c }) {
  const all = c.models
  const [preset, setPreset] = useState('all')
  const [custom, setCustom] = useState(all.map((r) => r.key))
  const def = PRESETS.find((p) => p[0] === preset)[2]
  let keys = all.map((r) => r.key)
  if (Array.isArray(def)) keys = def
  else if (def === 'ens_best') keys = ['ensemble', c.best_single]
  else if (def === 'custom') keys = custom
  const rows = all.filter((r) => keys.includes(r.key))
  const toggle = (k) => setCustom((cur) => (cur.includes(k) ? cur.filter((x) => x !== k) : [...cur, k]))

  const picker = (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '.8rem', alignItems: 'center', margin: '.5rem 0 1rem' }}>
      <label>Compare:{' '}
        <select value={preset} onChange={(e) => setPreset(e.target.value)}>
          {PRESETS.map(([k, l]) => <option key={k} value={k}>{l}</option>)}
        </select>
      </label>
      {preset === 'custom' && all.map((r) => (
        <label key={r.key} style={{ whiteSpace: 'nowrap' }}>
          <input type="checkbox" checked={custom.includes(r.key)} onChange={() => toggle(r.key)} /> {r.label}
        </label>
      ))}
    </div>
  )
  if (rows.length === 0) {
    return (
      <section className="panel">
        <h2>Detection models: accuracy, ROC curve and AUC</h2>
        {picker}
        <p className="muted">Select at least one model.</p>
      </section>
    )
  }
  const bestOf = Object.fromEntries(COLS.map(([k]) => [k, Math.max(...rows.map((r) => r[k]))]))
  const bestRow = rows.reduce((a, b) => ((b.f1 > a.f1 || (b.f1 === a.f1 && b.auc > a.auc)) ? b : a))
  const best = bestRow
  const cm = best.confusion_matrix
  return (
    <section className="panel">
      <h2>Detection models: accuracy, ROC curve and AUC</h2>
      <p className="muted small">{c.dataset}: {c.images} images, {(c.pixels / 1e6).toFixed(1)} million pixels. Bold = best in each column; "Best" = highest oil F1 among the selected models. Run on {c.created.slice(0, 10)}.</p>
      {picker}
      <table className="table">
        <thead><tr><th>Model</th><th>Backbone</th>{COLS.map(([k, l]) => <th key={k}>{l}</th>)}<th>Size (M params)</th><th>Speed (ms / image)</th></tr></thead>
        <tbody>{rows.map((r) => (
          <tr key={r.key} style={r.key === bestRow.key ? { background: '#FDF3DF' } : undefined}>
            <td>{r.label}{r.key === bestRow.key && <span className="badge pri-medium" style={{ marginLeft: '.4rem' }}>Best</span>}</td>
            <td>{r.encoder}</td>
            {COLS.map(([k]) => <td key={k} style={r[k] === bestOf[k] ? { fontWeight: 700 } : undefined}>{fmtNum(r[k], 3)}</td>)}
            <td>{fmtNum(r.params_m, 1)}</td><td>{fmtNum(r.ms_per_image, 0)}</td>
          </tr>
        ))}</tbody>
      </table>

      <div className="grid-2" style={{ marginTop: '1rem' }}>
        <div>
          <h3>ROC curve (oil vs sea, per pixel)</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart margin={{ top: 5, right: 10, bottom: 20, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="fpr" type="number" domain={[0, 1]} fontSize={11} label={{ value: 'False positive rate', position: 'insideBottom', offset: -10, fontSize: 11 }} />
              <YAxis dataKey="tpr" type="number" domain={[0, 1]} fontSize={11} label={{ value: 'True positive rate', angle: -90, position: 'insideLeft', fontSize: 11 }} />
              <Tooltip formatter={(v) => Number(v).toFixed(3)} />
              <Legend verticalAlign="top" />
              <Line data={[{ fpr: 0, tpr: 0 }, { fpr: 1, tpr: 1 }]} dataKey="tpr" name="Random (AUC 0.5)" stroke="#B0B4BA" strokeDasharray="5 5" dot={false} />
              {rows.map((r) => (
                <Line key={r.key} data={r.roc.fpr.map((f, i) => ({ fpr: f, tpr: r.roc.tpr[i] }))} dataKey="tpr"
                  name={`${r.label} (AUC ${fmtNum(r.auc, 3)})`} stroke={COLORS[r.key] || '#333'} strokeWidth={r.key === bestRow.key ? 3 : 1.5} dot={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div>
          <h3>Accuracy, F1 and AUC</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={rows} margin={{ top: 5, right: 10, bottom: 20, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="label" fontSize={10} interval={0} angle={-15} textAnchor="end" height={50} />
              <YAxis domain={[0.5, 1]} fontSize={11} />
              <Tooltip formatter={(v) => Number(v).toFixed(3)} /><Legend verticalAlign="top" />
              <Bar dataKey="accuracy" name="Accuracy" fill="#2C6E91" />
              <Bar dataKey="f1" name="F1 (oil)" fill="#C8322B" />
              <Bar dataKey="auc" name="ROC-AUC" fill="#E8A42A" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <h3 style={{ marginTop: '1rem' }}>Confusion matrix: {best.label} (pixels)</h3>
      <table className="table" style={{ maxWidth: 520 }}>
        <thead><tr><th /><th>Predicted sea</th><th>Predicted oil</th></tr></thead>
        <tbody>
          <tr><th>Actual sea</th><td>{cm[0][0].toLocaleString()} (true negative)</td><td>{cm[0][1].toLocaleString()} (false alarm)</td></tr>
          <tr><th>Actual oil</th><td>{cm[1][0].toLocaleString()} (missed oil)</td><td>{cm[1][1].toLocaleString()} (true positive)</td></tr>
        </tbody>
      </table>

      <Why c={c} />
    </section>
  )
}
