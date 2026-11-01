import { useEffect, useState } from 'react'
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import api, { errorText } from '../api/client'
import { Empty, ErrorNote, fmtNum } from '../components/Bits.jsx'
import FeedbackPanel from '../components/FeedbackPanel.jsx'
import ModelComparison from '../components/ModelComparison.jsx'

export default function ModelPerformance() {
  const [m, setM] = useState(null)
  const [error, setError] = useState('')
  useEffect(() => { api.get('/metrics').then((r) => setM(r.data)).catch((e) => setError(errorText(e))) }, [])
  if (error) return <ErrorNote text={error} />
  if (!m) return <div className="center-note">Loading…</div>

  const runs = m.segmentation_runs
  const testEntries = Object.entries(m.test_results)
  const latestTest = testEntries.length ? testEntries[testEntries.length - 1][1] : null
  const histories = Object.entries(m.histories)
  const attr = m.attribution
  const hasLookalike = runs.some((r) => r.lookalike_iou != null)

  return (
    <div className="page">
      <header className="page-head"><div><h1>Model performance</h1><p className="muted">Results written by the training and evaluation scripts.</p></div></header>

      <section className="panel">
        <h2>Oil spill detection: model comparison</h2>
        {runs.length === 0 ? (
          <Empty title="No trained models yet">Train on Colab with <code>python -m ml.segmentation.train --arch unet</code>, then copy the <code>models</code> folder back here. Until then the system uses the adaptive-threshold baseline.</Empty>
        ) : (
          <>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={runs}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="run_id" fontSize={11} />
                <YAxis domain={[0, 1]} fontSize={11} />
                <Tooltip /><Legend />
                <Bar dataKey="mIoU" fill="#2C6E91" name="Mean IoU" />
                <Bar dataKey="oil_iou" fill="#C8322B" name="Oil IoU" />
                {hasLookalike && <Bar dataKey="lookalike_iou" fill="#E8A42A" name="Look-alike IoU" />}
              </BarChart>
            </ResponsiveContainer>
            <table className="table">
              <thead><tr><th>Run</th><th>Model</th><th>Encoder</th><th>Mean IoU</th><th>Oil IoU</th>{hasLookalike && <th>Look-alike IoU</th>}<th>Oil precision</th><th>Oil recall</th><th>Oil F1</th></tr></thead>
              <tbody>{runs.map((r) => (
                <tr key={r.run_id}><td>{r.run_id}</td><td>{r.arch}</td><td>{r.arch === 'fcn' ? '–' : r.encoder}</td><td>{fmtNum(r.mIoU, 3)}</td><td>{fmtNum(r.oil_iou, 3)}</td>
                  {hasLookalike && <td>{fmtNum(r.lookalike_iou, 3)}</td>}<td>{fmtNum(r.oil_precision, 3)}</td><td>{fmtNum(r.oil_recall, 3)}</td><td>{fmtNum(r.oil_f1, 3)}</td></tr>
              ))}</tbody>
            </table>
          </>
        )}
      </section>

      <ModelComparison c={m.comparison} />

      {latestTest && (
        <section className="panel">
          <h2>Per-class IoU on the test set ({latestTest.weights})</h2>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={Object.entries(latestTest.per_class_iou).map(([k, v]) => ({ cls: k, iou: v }))}>
              <XAxis dataKey="cls" fontSize={11} /><YAxis domain={[0, 1]} fontSize={11} /><Tooltip />
              <Bar dataKey="iou" fill="#0F2A3D" name="IoU" />
            </BarChart>
          </ResponsiveContainer>
        </section>
      )}

      {histories.length > 0 && (
        <section className="panel">
          <h2>Training curves (validation oil IoU)</h2>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="epoch" type="number" fontSize={11} allowDuplicatedCategory={false} />
              <YAxis domain={[0, 1]} fontSize={11} /><Tooltip /><Legend />
              {histories.map(([run, h], i) => (
                <Line key={run} data={h} dataKey="oil_iou" name={run} dot={false} stroke={['#C8322B', '#2C6E91', '#E8A42A', '#3E8E6A'][i % 4]} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </section>
      )}

      <section className="panel">
        <h2>Vessel attribution accuracy (synthetic test cases)</h2>
        {!attr ? (
          <Empty title="Not evaluated yet">Run <code>python -m ml.attribution.synthetic_generator</code> and then <code>python -m ml.attribution.train_ranker</code>.</Empty>
        ) : (
          <>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={[{ m: 'Top-1', Rule: attr.rule_top1, ML: attr.ml_top1, Att: attr.attention_top1, Lgb: attr.lgbm_top1 }, { m: 'Top-3', Rule: attr.rule_top3, ML: attr.ml_top3, Att: attr.attention_top3, Lgb: attr.lgbm_top3 }]}>
                <XAxis dataKey="m" /><YAxis domain={[0, 1]} /><Tooltip /><Legend />
                <Bar dataKey="Rule" fill="#7A8C99" name="Rule-based score" />
                <Bar dataKey="ML" fill="#2C6E91" name={attr.model} />
                {attr.lgbm_top1 != null && <Bar dataKey="Lgb" fill="#3E8E6A" name="LightGBM LambdaRank" />}
                {attr.attention_top1 != null && <Bar dataKey="Att" fill="#6B2FA0" name="Cross-attention" />}
              </BarChart>
            </ResponsiveContainer>
            <table className="table">
              <thead><tr><th>Ranker</th><th>Top-1</th><th>Top-3</th></tr></thead>
              <tbody>
                <tr><td>Rule-based score</td><td>{(attr.rule_top1 * 100).toFixed(0)}%</td><td>{(attr.rule_top3 * 100).toFixed(0)}%</td></tr>
                <tr><td>{attr.model} (deployed)</td><td>{(attr.ml_top1 * 100).toFixed(0)}%</td><td>{(attr.ml_top3 * 100).toFixed(0)}%</td></tr>
                {attr.lgbm_top1 != null && <tr><td>LightGBM LambdaRank</td><td>{(attr.lgbm_top1 * 100).toFixed(0)}%</td><td>{(attr.lgbm_top3 * 100).toFixed(0)}%</td></tr>}
                {attr.attention_top1 != null && <tr><td>Cross-attention</td><td>{(attr.attention_top1 * 100).toFixed(0)}%</td><td>{(attr.attention_top3 * 100).toFixed(0)}%</td></tr>}
              </tbody>
            </table>
            <p className="muted small">{attr.n_test_cases} test cases out of {attr.n_cases}. The true vessel was among the candidates in {(attr.candidate_recall * 100).toFixed(0)}% of cases.</p>
          </>
        )}
      </section>

      {m.lstm && (
        <section className="panel">
          <h2>LSTM drift model (held-out simulated trajectories)</h2>
          <table className="table">
            <thead><tr><th>Forecast</th><th>LSTM error</th><th>Persistence baseline error</th></tr></thead>
            <tbody>{Object.keys(m.lstm.lstm_error_km).map((k) => (
              <tr key={k}><td>{k}</td><td>{m.lstm.lstm_error_km[k]} km</td><td>{m.lstm.persistence_error_km[k]} km</td></tr>
            ))}</tbody>
          </table>
          <p className="muted small">{m.lstm.model}. Forcing source: {m.lstm.forcing_source}. {m.lstm.test_sequences} test trajectories.</p>
        </section>
      )}

      {m.forecast_validation && (() => {
        const v = m.forecast_validation
        return (
          <section className="panel">
            <h2>Forecast check against a second satellite image</h2>
            <p className="muted small">Forecast made from the image of {new Date(v.first_time).toUTCString()}, compared with the oil detected in the image taken {v.gap_hours} h later ({v.second_image}). Wind in the second image: {v.second_image_wind_ms} m/s.</p>
            <table className="table">
              <thead><tr><th>Spill</th><th>Area</th><th>Predicted movement</th><th>Second image covers zone</th><th>Distance to nearest slick: forecast</th><th>LSTM</th><th>No-movement baseline</th><th>Oil inside forecast zone</th><th>Same slick?</th></tr></thead>
              <tbody>{v.spills.map((r) => (
                <tr key={r.spill_no}><td>#{r.spill_no}</td><td>{r.area_km2} km²</td><td>{r.moved_km} km</td>
                  <td>{(r.second_image_covers_zone * 100).toFixed(0)}%</td>
                  <td>{r.dist_forecast_km ?? '–'} km</td><td>{r.dist_lstm_km ?? '–'} km</td><td>{r.dist_persistence_km ?? '–'} km</td>
                  <td>{r.oil_in_forecast_zone_km2} km²</td><td>{r.matched ? 'Yes' : 'No'}</td></tr>
              ))}</tbody>
            </table>
            <p className={v.second_image_reliable ? '' : 'warn-note'}>{v.verdict}</p>
          </section>
        )
      })()}

      {m.drifter_validation && (() => {
        const d = m.drifter_validation
        const rows = [['current_only', 'Drift model (current only)'], ['current_plus_1pct_wind', 'Drift model (current + 1% wind)'], ['no_movement', 'No-movement baseline']]
        return (
          <section className="panel">
            <h2>Drift model vs real ocean drifters</h2>
            <p className="muted small">{d.segments_used} pieces of 72-hour tracks from {d.drifters} NOAA drifters, region {d.region.join(', ')}, {d.period.join(' to ')}. The model starts at each drifter's first position; the table shows how far it ends up from where the drifter really was.</p>
            <table className="table">
              <thead><tr><th>Forecast</th><th>after 24 h</th><th>after 48 h</th><th>after 72 h</th><th>Skill score</th></tr></thead>
              <tbody>{rows.map(([k, label]) => (
                <tr key={k}><td>{label}</td>{['24h', '48h', '72h'].map((h) => <td key={h}>{d.mean_separation_km[k]?.[h] ?? '–'} km</td>)}<td>{d.skill[k] ?? '–'}</td></tr>
              ))}</tbody>
            </table>
            <p className="muted small">Real drifters travelled on average {d.mean_real_path_km['24h']} / {d.mean_real_path_km['48h']} / {d.mean_real_path_km['72h']} km in 24 / 48 / 72 h. Skill score: 0 = no skill, 1 = perfect (Liu &amp; Weisberg 2011).</p>
            <p>{d.verdict}</p>
          </section>
        )
      })()}

      {m.lookalike && (() => {
        const l = m.lookalike
        const rows = [['rule_based', 'Rules only (before)'], ['logistic_regression', 'Logistic regression'], ['gradient_boosting', 'Gradient boosting']]
        const imp = Object.entries(l.feature_importance || {}).slice(0, 5)
        return (
          <section className="panel">
            <h2>Look-alike model (oil vs look-alike)</h2>
            <p className="muted small">Trained on {l.samples.oil} expert-labelled oil slicks and {l.samples.lookalike} look-alike patches from {l.samples.passes} Sentinel-1 passes ({l.dataset}). Tested on satellite passes the model never saw ({l.validation}).</p>
            <table className="table">
              <thead><tr><th>Method</th><th>ROC-AUC</th><th>Accuracy</th><th>F1 (oil)</th></tr></thead>
              <tbody>{rows.filter(([k]) => l.results[k]).map(([k, label]) => (
                <tr key={k}><td>{label}{k === l.chosen ? ' ✓ chosen' : ''}</td><td>{l.results[k].auc}</td><td>{l.results[k].accuracy}</td><td>{l.results[k].f1_oil}</td></tr>
              ))}</tbody>
            </table>
            <p className="muted small">Most useful measurements: {imp.map(([k, v]) => `${k.replace(/_/g, ' ')} (${v})`).join(', ')}.</p>
            <p className="muted small">Limitation: on the confirmed Huntington Beach spill (California, 2021) this Mediterranean-trained model rated the real slicks as look-alikes, so it is shown in each incident for information only and does not change the oil score. Training on images from more regions is future work.</p>
          </section>
        )
      })()}

      <FeedbackPanel />

      {m.figures.length > 0 && (
        <section className="panel">
          <h2>Sample predictions (input, ground truth, prediction)</h2>
          <div className="gallery">{m.figures.map((f) => <img key={f} src={`/figures/${f}`} alt={f} loading="lazy" />)}</div>
        </section>
      )}
    </div>
  )
}
