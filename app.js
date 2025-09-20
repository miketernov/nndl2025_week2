/* Titanic TF.js — Browser-only Binary Classifier
   ------------------------------------------------
   Data schema (reuse-friendly):
   - Target: Survived (0/1)
   - Features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
   - Identifier: PassengerId (exclude from training)
   // REUSE NOTE: To adapt for another dataset, swap the SCHEMA, update preprocessing, rebuild one-hots, and keep the UI.
*/

(() => {
  // ======= Global State =======
  const state = {
    rawTrain: null,     // array of row objects (train)
    rawTest: null,      // array of row objects (test)
    previewHeaders: [], // cached column order for preview
    xTrain: null, yTrain: null,
    xVal: null, yVal: null,
    xTest: null,
    trainIndex: null, valIndex: null, // original index mapping after split
    catMaps: { Sex: [], Pclass: [], Embarked: [] }, // category lists for one-hot
    scalers: { Age: {mean:0, std:1}, Fare: {mean:0, std:1} },
    imputes: { Age: null, Embarked: null },
    addFamily: true,
    model: null,
    valProbs: null, valLabels: null,
    auc: null,
    youdenThreshold: null,
    testProbs: null,
    submission: null, probsCSV: null,
  };

  // ======= DOM Helpers =======
  const $ = sel => document.querySelector(sel);
  const setHTML = (sel, html) => ($(sel).innerHTML = html);
  const appendHTML = (sel, html) => ($(sel).innerHTML += html);
  const fmt = (x, d=4) => (Number.isFinite(x) ? Number(x).toFixed(d) : String(x));

  // ======= CSV Parsing (robust enough for Kaggle Titanic with quoted Name) =======
  // Parses CSV text into array of objects. Keeps header names. Handles quotes and commas in quotes.
  function parseCSV(text){
    const rows = [];
    const headers = [];
    let i = 0, field = '', row = [], inQuotes = false;

    const pushField = () => {
      row.push(inQuotes ? field.replace(/""/g, '"') : field);
      field = '';
    };
    const pushRow = () => {
      if (headers.length === 0) headers.push(...row);
      else {
        const obj = {};
        for (let c=0;c<headers.length;c++) obj[headers[c]] = row[c] ?? '';
        rows.push(obj);
      }
      row = [];
    };

    while (i < text.length){
      const ch = text[i];
      if (inQuotes){
        if (ch === '"'){
          if (text[i+1] === '"'){ field += '"'; i++; } // escaped quote
          else inQuotes = false;
        } else field += ch;
      } else {
        if (ch === '"') inQuotes = true;
        else if (ch === ',') pushField();
        else if (ch === '\n'){
          pushField(); pushRow();
        } else if (ch === '\r'){
          // handle CRLF -> skip; row will end on \n
        } else field += ch;
      }
      i++;
    }
    // flush last field/row if file didn't end with newline
    if (field.length || row.length){
      pushField(); pushRow();
    }
    return rows;
  }

  // ======= Utility: download string/binary as file =======
  function downloadFile(filename, content, mime='text/csv'){
    const blob = new Blob([content], {type: mime});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = filename; document.body.appendChild(a); a.click();
    a.remove(); URL.revokeObjectURL(url);
  }

  // ======= Data helpers =======
  const SCHEMA = {
    target: 'Survived',
    id: 'PassengerId',
    // source columns we care about (others are ignored if present)
    columns: ['PassengerId','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
  };

  function pickAndCoerce(row, isTrain){
    // Copy only needed columns (string -> number where applicable).
    const obj = {};
    for (const col of SCHEMA.columns){
      obj[col] = row[col] ?? '';
    }
    // Numeric coercions
    for (const k of ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']){
      if (k in obj && obj[k] !== '') obj[k] = +obj[k];
      else if (k in obj) obj[k] = null;
    }
    // Categorical keep as string (Sex, Embarked, Pclass will be one-hot; Pclass also numeric but we one-hot it)
    return obj;
  }

  function computeMissingness(rows){
    const cols = Object.keys(rows[0] || {});
    const n = rows.length || 0;
    const miss = {};
    for (const c of cols) miss[c] = 0;
    for (const r of rows){
      for (const c of cols){
        const v = r[c];
        if (v === '' || v === null || v === undefined || (typeof v === 'number' && !Number.isFinite(v))) miss[c]++;
      }
    }
    return { n, per: Object.fromEntries(cols.map(c => [c, n ? (miss[c]/n)*100 : 0])) };
  }

  function makePreview(rows, max=8){
    if (!rows || rows.length === 0) return '<em class="muted">No data.</em>';
    const cols = Object.keys(rows[0]);
    let html = '<table><thead><tr>';
    for (const c of cols) html += `<th>${c}</th>`;
    html += '</tr></thead><tbody>';
    const m = Math.min(max, rows.length);
    for (let i=0;i<m;i++){
      html += '<tr>';
      for (const c of cols) html += `<td>${rows[i][c]}</td>`;
      html += '</tr>';
    }
    if (rows.length > m){
      html += `<tr><td colspan="${cols.length}" class="muted">… ${rows.length - m} more rows</td></tr>`;
    }
    html += '</tbody></table>';
    return html;
  }

  // ======= Simple charts with tfjs-vis =======
  async function plotBar(containerId, seriesName, categories, values, title, xLabel, yLabel){
    const data = categories.map((c,i)=>({x: String(c), y: values[i]}));
    await tfvis.render.barchart(
      {name: title, tab:'EDA', styles:{height:'320px'}, drawArea: document.getElementById(containerId)},
      data, { xLabel, yLabel, series: [seriesName] }
    );
  }

  // ======= EDA: Survival by Sex/Pclass =======
  function survivalRates(rows){
    const by = (key) => {
      const map = {}; // value -> [sum, count]
      for (const r of rows){
        const k = r[key];
        if (!(k in map)) map[k] = [0,0];
        const y = +r[SCHEMA.target];
        if (Number.isFinite(y)) { map[k][0] += y; map[k][1] += 1; }
      }
      const cats = Object.keys(map);
      const rates = cats.map(k => map[k][1] ? map[k][0]/map[k][1] : 0);
      return { cats, rates };
    };
    return {
      sex: by('Sex'),
      pclass: by('Pclass')
    };
  }

  // ======= Preprocessing =======
  function mode(arr){
    const freq = new Map();
    for (const v of arr){
      if (v==='' || v==null) continue;
      freq.set(v, (freq.get(v)||0)+1);
    }
    let best=null, bestC=-1;
    for (const [k,c] of freq.entries()){
      if (c>bestC){best=k;bestC=c;}
    }
    return best ?? null;
  }
  const median = arr => {
    const a = arr.filter(v=>Number.isFinite(v)).sort((a,b)=>a-b);
    if (!a.length) return null;
    const mid = Math.floor(a.length/2);
    return a.length%2 ? a[mid] : (a[mid-1]+a[mid])/2;
  };
  const meanStd = arr => {
    const a = arr.filter(v=>Number.isFinite(v));
    const m = a.reduce((s,v)=>s+v,0)/(a.length||1);
    const sd = Math.sqrt(a.reduce((s,v)=>s+(v-m)*(v-m),0)/(a.length||1)) || 1;
    return {mean:m, std:sd};
  };

  function buildCategoryMap(values){
    // stable order
    const set = new Set();
    for (const v of values){
      if (v!=='' && v!=null) set.add(String(v));
    }
    return Array.from(set);
  }

  function oneHot(value, cats){
    const idx = cats.indexOf(String(value));
    return cats.map((_,i)=> i===idx ? 1 : 0);
  }

  function standardize(v, {mean,std}, fallback=0){
    if (v===null || v==='' || !Number.isFinite(v)) return fallback;
    return (v - mean) / (std || 1);
  }

  function buildFeatures(rows, cfg){
    // cfg: {catMaps, scalers, imputes, addFamily}
    const X = [];
    const Y = [];
    for (const r of rows){
      // Impute
      const age = (r.Age==null || !Number.isFinite(r.Age)) ? cfg.imputes.Age : r.Age;
      const embarked = (r.Embarked==='' || r.Embarked==null) ? cfg.imputes.Embarked : r.Embarked;

      // Optional engineered
      const familySize = (Number.isFinite(r.SibSp)?r.SibSp:0) + (Number.isFinite(r.Parch)?r.Parch:0) + 1;
      const isAlone = familySize===1 ? 1 : 0;

      // Continuous standardized
      const ageZ = standardize(age, cfg.scalers.Age, 0);
      const fareZ = standardize(r.Fare, cfg.scalers.Fare, 0);

      // One-hots
      const ohSex = oneHot(r.Sex, cfg.catMaps.Sex);
      const ohPclass = oneHot(r.Pclass, cfg.catMaps.Pclass);
      const ohEmb = oneHot(embarked, cfg.catMaps.Embarked);

      const base = [
        ageZ, fareZ,
        Number.isFinite(r.SibSp)?r.SibSp:0,
        Number.isFinite(r.Parch)?r.Parch:0,
        ...ohSex, ...ohPclass, ...ohEmb
      ];
      const features = cfg.addFamily ? [...base, familySize, isAlone] : base;
      X.push(features);

      if ('Survived' in r && r.Survived !== '' && r.Survived !== null){
        Y.push(+r.Survived);
      }
    }
    const xTensor = tf.tensor2d(X);
    const yTensor = Y.length ? tf.tensor2d(Y, [Y.length, 1]) : null;
    return { xTensor, yTensor, featureCount: X[0]?.length || 0 };
  }

  // Stratified 80/20 split
  function stratifiedSplit(rows, yKey='Survived', valRatio=0.2){
    const pos = [], neg = [];
    rows.forEach((r, idx) => {
      const y = +r[yKey];
      if (y===1) pos.push(idx); else neg.push(idx);
    });
    const shuffle = arr => arr.sort(()=>Math.random()-0.5);
    shuffle(pos); shuffle(neg);

    const vp = Math.floor(pos.length*valRatio);
    const vn = Math.floor(neg.length*valRatio);
    const valIdx = new Set([...pos.slice(0,vp), ...neg.slice(0,vn)]);
    const trainIdx = rows.map((_,i)=>i).filter(i=>!valIdx.has(i));

    return { trainIdx, valIdx: Array.from(valIdx) };
  }

  function subsetByIndex(arr, idxs){
    return idxs.map(i=>arr[i]);
  }

  // ======= Model =======
  function buildModel(inputDim){
    const model = tf.sequential();
    model.add(tf.layers.dense({units:16, activation:'relu', inputShape:[inputDim]}));
    model.add(tf.layers.dense({units:1, activation:'sigmoid'}));
    model.compile({
      optimizer:'adam',
      loss:'binaryCrossentropy',
      metrics:['accuracy']
    });
    return model;
  }

  function summarizeModel(model){
    const lines = [];
    model.summary(80, undefined, line => lines.push(line));
    return lines.join('\n');
  }

  // ======= ROC/AUC & Threshold metrics =======
  function computeROC(probs, labels){
    const pairs = probs.map((p,i)=>({p, y: labels[i]})).sort((a,b)=>b.p - a.p);
    const P = labels.reduce((s,y)=>s+(y===1), 0);
    const N = labels.length - P;

    const roc = [];
    let tp = 0, fp = 0, prevP = Infinity;
    for (let i=0;i<pairs.length;i++){
      const {p, y} = pairs[i];
      if (p !== prevP){
        // record point at previous threshold
        if (i>0){
          roc.push({ fpr: fp/(N||1), tpr: tp/(P||1), thr: prevP });
        }
        prevP = p;
      }
      if (y===1) tp++; else fp++;
    }
    // last point
    roc.push({ fpr: fp/(N||1), tpr: tp/(P||1), thr: pairs[pairs.length-1]?.p ?? 0 });
    // Ensure (0,0) and (1,1) anchors
    roc.unshift({fpr:0, tpr:0, thr:1});
    roc.push({fpr:1, tpr:1, thr:0});

    // Trapezoidal AUC
    let auc = 0;
    for (let i=1;i<roc.length;i++){
      const x1 = roc[i-1].fpr, y1 = roc[i-1].tpr;
      const x2 = roc[i].fpr, y2 = roc[i].tpr;
      auc += (x2 - x1) * (y1 + y2) / 2;
    }

    // Youden J best threshold
    let bestJ = -Infinity, bestThr = 0.5;
    for (const pt of roc){
      const J = (pt.tpr) - (pt.fpr);
      if (J > bestJ){ bestJ = J; bestThr = pt.thr; }
    }
    return { roc, auc, youden: bestThr };
  }

  function confusionAtThreshold(probs, labels, thr){
    let TP=0, FP=0, TN=0, FN=0;
    for (let i=0;i<probs.length;i++){
      const pred = probs[i] >= thr ? 1 : 0;
      const y = labels[i];
      if (pred===1 && y===1) TP++;
      else if (pred===1 && y===0) FP++;
      else if (pred===0 && y===0) TN++;
      else FN++;
    }
    const precision = (TP+FP) ? TP/(TP+FP) : 0;
    const recall = (TP+FN) ? TP/(TP+FN) : 0;
    const f1 = (precision+recall) ? (2*precision*recall)/(precision+recall) : 0;
    return {TP,FP,TN,FN, precision, recall, f1};
  }

  async function plotROC(roc, auc){
    const points = roc.map(pt => ({x: pt.fpr, y: pt.tpr}));
    await tfvis.render.linechart(
      {name:'ROC Curve', tab:'Metrics', styles:{height:'320px'}, drawArea: $('#rocPlot')},
      {values: points, series: ['ROC']},
      {xLabel:'FPR', yLabel:'TPR', xAxisDomain:[0,1], yAxisDomain:[0,1], zoomToFit:true}
    );
    $('#mAUC').textContent = fmt(auc, 4);
  }

  // ======= UI Actions =======
  async function onLoad(){
    try{
      const trainFile = $('#trainFile').files[0];
      const testFile = $('#testFile').files[0];
      if (!trainFile){ alert('Please select train.csv'); return; }
      if (!testFile){ alert('Please select test.csv'); return; }

      const [trainText, testText] = await Promise.all([trainFile.text(), testFile.text()]);
      const trainRows0 = parseCSV(trainText).map(r => pickAndCoerce(r, true));
      const testRows0  = parseCSV(testText).map(r => pickAndCoerce(r, false));

      // Basic checks
      const reqTrainCols = ['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','PassengerId'];
      const reqTestCols = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','PassengerId'];
      const hasAll = (row, cols) => cols.every(c => Object.prototype.hasOwnProperty.call(row, c));
      if (!trainRows0.length || !hasAll(trainRows0[0], reqTrainCols)){ alert('Train CSV missing required columns.'); return; }
      if (!testRows0.length || !hasAll(testRows0[0], reqTestCols)){ alert('Test CSV missing required columns.'); return; }

      state.rawTrain = trainRows0;
      state.rawTest = testRows0;
      state.previewHeaders = Object.keys(trainRows0[0]);

      // Preview & missingness
      setHTML('#previewTable', makePreview(trainRows0));
      const miss = computeMissingness(trainRows0);
      const missHTML = Object.entries(miss.per).map(([k,v])=>`<div class="kv"><div>${k}</div><div>${fmt(v,2)}%</div></div>`).join('');
      setHTML('#missingSummary', `<div class="kv"><div><strong>Rows</strong></div><div>${miss.n}</div></div>${missHTML}`);

      // EDA plots
      const surv = survivalRates(trainRows0);
      await plotBar('chart-sex', 'Survival Rate', surv.sex.cats, surv.sex.rates, 'Survival by Sex', 'Sex', 'Rate');
      await plotBar('chart-pclass', 'Survival Rate', surv.pclass.cats, surv.pclass.rates, 'Survival by Pclass', 'Pclass', 'Rate');

      setHTML('#loadSummary', `
        <div class="kv"><div>Train rows</div><div>${trainRows0.length}</div></div>
        <div class="kv"><div>Test rows</div><div>${testRows0.length}</div></div>
      `);
    }catch(err){
      console.error(err);
      alert('Failed to load/parse CSVs. Please ensure they are the original Kaggle files.');
    }
  }

  function onReset(){
    // Dispose tensors/model if present
    for (const t of ['xTrain','yTrain','xVal','yVal','xTest']){
      if (state[t]){ state[t].dispose(); state[t] = null; }
    }
    if (state.model){ state.model.dispose(); state.model = null; }
    Object.assign(state, {
      valProbs:null, valLabels:null, auc:null, youdenThreshold:null,
      testProbs:null, submission:null, probsCSV:null
    });
    // Clear UI
    setHTML('#loadSummary','');
    setHTML('#previewTable','');
    setHTML('#missingSummary','');
    setHTML('#preSummary','');
    setHTML('#modelSummary','');
    setHTML('#trainPlots','');
    setHTML('#rocPlot','');
    $('#mTP').textContent = $('#mFP').textContent = $('#mTN').textContent = $('#mFN').textContent = '–';
    $('#mPrec').textContent = $('#mRec').textContent = $('#mF1').textContent = $('#mAUC').textContent = '–';
    $('#thresholdSlider').value = 0.5; $('#thresholdVal').textContent = '0.50';
  }

  function onPreprocess(){
    try{
      if (!state.rawTrain || !state.rawTest){ alert('Load data first.'); return; }
      state.addFamily = $('#chkFamily').checked;

      // Fit imputations on TRAIN ONLY
      state.imputes.Age = median(state.rawTrain.map(r=>r.Age));
      state.imputes.Embarked = mode(state.rawTrain.map(r=>r.Embarked));

      // Fit scalers on TRAIN ONLY (Age, Fare)
      state.scalers.Age = meanStd(state.rawTrain.map(r=>{
        const v = (r.Age==null||!Number.isFinite(r.Age)) ? state.imputes.Age : r.Age;
        return v;
      }));
      state.scalers.Fare = meanStd(state.rawTrain.map(r=>r.Fare));

      // Build category maps from TRAIN ONLY (stable)
      state.catMaps.Sex = buildCategoryMap(state.rawTrain.map(r=>r.Sex));         // typically ["male","female"]
      state.catMaps.Pclass = buildCategoryMap(state.rawTrain.map(r=>r.Pclass));   // ["3","1","2"] etc
      state.catMaps.Embarked = buildCategoryMap(state.rawTrain.map(r=>r.Embarked)); // ["S","C","Q"]

      // Split indices (stratified)
      const {trainIdx, valIdx} = stratifiedSplit(state.rawTrain, SCHEMA.target, 0.2);
      state.trainIndex = trainIdx; state.valIndex = valIdx;

      const trainRows = subsetByIndex(state.rawTrain, trainIdx);
      const valRows   = subsetByIndex(state.rawTrain, valIdx);

      const cfg = { catMaps:state.catMaps, scalers:state.scalers, imputes:state.imputes, addFamily:state.addFamily };
      const {xTensor:xTr, yTensor:yTr, featureCount:f1} = buildFeatures(trainRows, cfg);
      const {xTensor:xVa, yTensor:yVa, featureCount:f2} = buildFeatures(valRows, cfg);
      const {xTensor:xTe} = buildFeatures(state.rawTest, cfg);

      // Save tensors (dispose previous if exist)
      for (const k of ['xTrain','yTrain','xVal','yVal','xTest']){
        if (state[k]) state[k].dispose();
      }
      state.xTrain = xTr; state.yTrain = yTr;
      state.xVal = xVa; state.yVal = yVa;
      state.xTest = xTe;

      const featureNames = [
        'Age_z','Fare_z','SibSp','Parch',
        ...state.catMaps.Sex.map(v=>`Sex=${v}`),
        ...state.catMaps.Pclass.map(v=>`Pclass=${v}`),
        ...state.catMaps.Embarked.map(v=>`Embarked=${v}`),
        ...(state.addFamily ? ['FamilySize','IsAlone'] : [])
      ];

      setHTML('#preSummary', `
        <div class="kv"><div>Train shape</div><div>${xTr.shape.join('×')} (features: ${featureNames.length})</div></div>
        <div class="kv"><div>Val shape</div><div>${xVa.shape.join('×')}</div></div>
        <div class="kv"><div>Test shape</div><div>${xTe.shape.join('×')}</div></div>
        <div class="kv"><div>Impute Age (median)</div><div>${fmt(state.imputes.Age,2)}</div></div>
        <div class="kv"><div>Impute Embarked (mode)</div><div>${state.imputes.Embarked}</div></div>
        <div class="kv"><div>Standardize Age</div><div>μ=${fmt(state.scalers.Age.mean,2)}, σ=${fmt(state.scalers.Age.std,2)}</div></div>
        <div class="kv"><div>Standardize Fare</div><div>μ=${fmt(state.scalers.Fare.mean,2)}, σ=${fmt(state.scalers.Fare.std,2)}</div></div>
        <div class="kv"><div>One-hot categories</div>
          <div>Sex=[${state.catMaps.Sex.join(', ')}]; Pclass=[${state.catMaps.Pclass.join(', ')}]; Embarked=[${state.catMaps.Embarked.join(', ')}]</div>
        </div>
      `);
    }catch(err){
      console.error(err);
      alert('Preprocessing failed. Check console for details.');
    }
  }

  function onBuild(){
    try{
      if (!state.xTrain) { alert('Run preprocessing first.'); return; }
      if (state.model) { state.model.dispose(); state.model = null; }
      state.model = buildModel(state.xTrain.shape[1]);
      setHTML('#modelSummary', `<span class="muted">Model built. Click "Show Summary".</span>`);
    }catch(err){
      console.error(err);
      alert('Model build failed.');
    }
  }

  function onSummary(){
    if (!state.model){ alert('Build the model first.'); return; }
    const text = summarizeModel(state.model);
    setHTML('#modelSummary', text);
  }

  async function onTrain(){
    try{
      if (!state.model) { alert('Build the model first.'); return; }
      if (!state.xTrain || !state.yTrain || !state.xVal || !state.yVal){ alert('Run preprocessing first.'); return; }

      const container = { name:'Training Curves', tab:'Training', styles:{ height:'350px' }, drawArea: $('#trainPlots') };
      const metrics = ['loss','val_loss','acc','val_acc','accuracy','val_accuracy']; // include alt keys
      const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

      const es = tf.callbacks.earlyStopping({ monitor: 'val_loss', patience: 5, restoreBestWeights: true });

      const history = await state.model.fit(state.xTrain, state.yTrain, {
        epochs: 50, batchSize: 32,
        validationData: [state.xVal, state.yVal],
        callbacks: [fitCallbacks, es],
        shuffle: true,
      });
      console.log('History:', history.history);

      // Store validation probs for ROC
      const valPred = state.model.predict(state.xVal);
      const probs = Array.from(await valPred.data());
      valPred.dispose();
      const labels = Array.from(await state.yVal.data());

      state.valProbs = probs; state.valLabels = labels;
      const {roc, auc, youden} = computeROC(probs, labels);
      state.auc = auc; state.youdenThreshold = youden;
      await plotROC(roc, auc);

      // Initialize threshold metrics
      $('#thresholdSlider').value = 0.5;
      $('#thresholdVal').textContent = '0.50';
      updateMetricsAtThreshold(0.5);

      $('#exportNote').textContent = 'After training, you can save the model (downloads both JSON and weights).';
    }catch(err){
      console.error(err);
      alert('Training failed. Check console for details.');
    }
  }

  function onEvaluate(){
    if (!state.valProbs || !state.valLabels){ alert('Train first to generate validation predictions.'); return; }
    // Re-plot ROC/metrics in case user cleared tab
    const {roc, auc, youden} = computeROC(state.valProbs, state.valLabels);
    state.auc = auc; state.youdenThreshold = youden;
    plotROC(roc, auc);
    updateMetricsAtThreshold(parseFloat($('#thresholdSlider').value || '0.5'));
  }

  function updateMetricsAtThreshold(thr){
    if (!state.valProbs || !state.valLabels) return;
    const {TP,FP,TN,FN, precision, recall, f1} = confusionAtThreshold(state.valProbs, state.valLabels, thr);
    $('#mTP').textContent = TP; $('#mFP').textContent = FP; $('#mTN').textContent = TN; $('#mFN').textContent = FN;
    $('#mPrec').textContent = fmt(precision,4);
    $('#mRec').textContent = fmt(recall,4);
    $('#mF1').textContent = fmt(f1,4);
  }

  async function onPredict(){
    try{
      if (!state.model){ alert('Train a model first.'); return; }
      if (!state.xTest){ alert('Load & preprocess test.csv first.'); return; }

      const thr = parseFloat($('#thresholdSlider').value || '0.5');

      const preds = state.model.predict(state.xTest);
      const probs = Array.from(await preds.data());
      preds.dispose();
      state.testProbs = probs;

      // Build submission.csv using current threshold
      const ids = state.rawTest.map(r=>r.PassengerId);
      const survived = probs.map(p => (p >= thr ? 1 : 0));
      const rows = [['PassengerId','Survived'], ...ids.map((id,i)=>[id, survived[i]])];
      const csv = rows.map(r=>r.join(',')).join('\n');
      state.submission = csv;

      // probabilities.csv
      const probRows = [['PassengerId','Probability'], ...ids.map((id,i)=>[id, probs[i]])];
      state.probsCSV = probRows.map(r=>r.join(',')).join('\n');

      setHTML('#predSummary', `
        <div class="kv"><div>Test rows</div><div>${ids.length}</div></div>
        <div class="kv"><div>Threshold</div><div>${fmt(thr,2)}</div></div>
        <div class="kv"><div>Predicted positives</div><div>${survived.reduce((s,v)=>s+v,0)}</div></div>
      `);
    }catch(err){
      console.error(err);
      alert('Prediction failed.');
    }
  }

  async function onSaveModel(){
    try{
      if (!state.model){ alert('No model to save.'); return; }
      await state.model.save('downloads://titanic-tfjs');
    }catch(err){
      console.error(err);
      alert('Saving model failed.');
    }
  }

  function onDownloadSubmission(){
    if (!state.submission){ alert('No submission built. Click "Predict Test" first.'); return; }
    downloadFile('submission.csv', state.submission, 'text/csv');
  }

  function onDownloadProbs(){
    if (!state.probsCSV){ alert('No probabilities available. Click "Predict Test" first.'); return; }
    downloadFile('probabilities.csv', state.probsCSV, 'text/csv');
  }

  // ======= Wire Events =======
  $('#btnLoad').addEventListener('click', onLoad);
  $('#btnReset').addEventListener('click', onReset);
  $('#btnPreprocess').addEventListener('click', onPreprocess);
  $('#btnBuild').addEventListener('click', onBuild);
  $('#btnSummary').addEventListener('click', onSummary);
  $('#btnTrain').addEventListener('click', onTrain);
  $('#btnEvaluate').addEventListener('click', onEvaluate);
  $('#btnPredict').addEventListener('click', onPredict);
  $('#btnSaveModel').addEventListener('click', onSaveModel);
  $('#btnDownloadSubmission').addEventListener('click', onDownloadSubmission);
  $('#btnDownloadProbs').addEventListener('click', onDownloadProbs);

  // Threshold slider live updates
  $('#thresholdSlider').addEventListener('input', (e)=>{
    const v = parseFloat(e.target.value);
    $('#thresholdVal').textContent = v.toFixed(2);
    updateMetricsAtThreshold(v);
  });

  // Quick buttons: default 0.5 and Youden
  $('#btnDefault05').addEventListener('click', ()=>{
    $('#thresholdSlider').value = 0.5; $('#thresholdVal').textContent = '0.50';
    updateMetricsAtThreshold(0.5);
  });
  $('#btnYouden').addEventListener('click', ()=>{
    if (state.youdenThreshold==null){ alert('Compute ROC first (train).'); return; }
    const t = Math.min(1, Math.max(0, state.youdenThreshold));
    $('#thresholdSlider').value = t; $('#thresholdVal').textContent = t.toFixed(2);
    updateMetricsAtThreshold(t);
  });

})();
