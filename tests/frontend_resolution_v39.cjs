// Run actual production sizing and facade code; emit previews for Python parity.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = process.env.H3_TEST_ROOT || path.resolve(__dirname, '..');
const graph = { links: {}, getNodeById() { return this.source; } };
const app = { graph, registerExtension() {}, ui: { settings: { getSettingValue(_id, fallback) { return fallback; } } } };
const sandbox = { app, api: { addEventListener() {} }, console, setTimeout(cb) { cb(); }, clearTimeout() {} };
const source = fs.readFileSync(path.join(root, 'web/project_id.js'), 'utf8').replace(/^import .*;\r?$/gm, '');
vm.runInNewContext(source + '\nglobalThis.f={configureResolutionPresetWidgets,configureIntuitiveV38Ux,installProductionSerializationGuard};', sandbox);
const f = sandbox.f;
const presets = ['Draft — 0.30 MP', 'Balanced — 0.60 MP', 'Native 768', 'Custom'];
const aspects = ['Auto from First Image','Square 1:1','4:3','3:2','16:10','Landscape 16:9','21:9','3:4','2:3','10:16','Portrait 9:16'];
const labels = ['Auto / First Image','1:1','4:3','3:2','16:10','16:9','21:9','3:4','2:3','10:16','9:16'];
const defaults = {aspect:'Auto from First Image',preset:presets[0],custom_mp:0.37,
    size_source:'Legacy Aspect',width:960,height:640, chunks:3,chunk_seconds:5,
    run_storage:'Off',generation_mode:'Full Run',reference_size:'Match Output',
    video_reference_size:'Efficient - 0.4 MP',regenerate_from_chunk:'Auto',
    base_seed:1,audio_continuity:true,continuity:'Balanced',prompt_mode:'Fixed'};
function make(inputs={}, geometry=null, linked=false, cls='H3ContinuumSamplerV39') {
    graph.links = linked || geometry ? {1:{origin_id:9}} : {};
    graph.source = {mode:0, imgs:geometry ? [{naturalWidth:geometry[0],naturalHeight:geometry[1]}] : []};
    const values = {...defaults,...inputs};
    const n={comfyClass:cls,properties:{},inputs:[{name:'first_frame',link:linked||geometry?1:null}],size:[500,600],
        widgets:Object.entries(values).map(([name,value])=>({name,value,type:'number',options:{},computeSize:()=>[120,20]})),
        addWidget(type,name,value,callback,options){const w={type,name,value,callback,options,computeSize:()=>[120,20]};this.widgets.push(w);return w;},
        addCustomWidget(w){this.widgets.push(w);return w;},setDirtyCanvas(){},
        serialize(){return {widgets_values:this.widgets.map(w=>w.value)};},
        configure(info){info.widgets_values.forEach((v,i)=>{if(this.widgets[i])this.widgets[i].value=v;});},
    };
    n.original = structuredClone(n.serialize());
    f.installProductionSerializationGuard(n);
    f.configureResolutionPresetWidgets(n);
    f.configureIntuitiveV38Ux(n);
    return n;
}
const widget=(n,name)=>n.widgets.find(w=>w.name===name);
function edit(n,name,value){const w=widget(n,name);assert(w,name);w.value=value;w.callback?.(value);}
const rows=[];
for(const aspect of aspects) for(const preset of presets) {
    const geometries=aspect===aspects[0] ? [[1200,800],[800,1200],[900,900],null] : [null];
    for(const geometry of geometries) for(const mp of preset==='Custom'?[0.1,0.37,2.5,16]:[0.37]) {
        const inputs={size_source:'Megapixel + Aspect Ratio',aspect,preset,custom_mp:mp,width:960,height:640};
        const n=make(inputs,geometry);
        assert.deepEqual(n.serialize(),n.original,'configure/preview changed serialized values');
        assert.deepEqual(Array.from(widget(n,'Aspect Ratio').options.values),labels);
        assert.deepEqual(Array.from(widget(n,'Resolution Mode').options.values),['Megapixel + Aspect Ratio','Manual Width × Height']);
        const preview=structuredClone(n.__h3ContinuumResolutionPreview());
        const display=widget(n,'Resolved Size');
        if(preview.pending) assert.match(display.detail,/Pending \/ unknown/);
        else assert(display.detail.includes(preview.actualMP.toFixed(6)+' MP'));
        assert.equal(display.options.serialize,false);
        rows.push({inputs,geometry,preview});
        // Actual serialize/configure cycle, including inactive manual values.
        const saved=structuredClone(n.serialize());
        const loaded=make({},geometry);
        loaded.configure(saved);
        loaded.__h3ContinuumIntuitiveUxRefresh();
        assert.deepEqual(loaded.serialize(),saved);
        assert.deepEqual(structuredClone(loaded.__h3ContinuumResolutionPreview()),preview);
        edit(n,'Resolution Mode','Manual Width × Height');
        assert.equal(widget(n,'aspect').value,aspect);
        assert.equal(widget(n,'preset').value,preset);
        assert.equal(widget(n,'custom_mp').value,mp);
        assert.equal(widget(n,'width').value,960);
        assert.equal(widget(n,'height').value,640);
        assert.notEqual(widget(n,'Width').hidden,true);
        assert.equal(widget(n,'Width').type,'number');
        edit(n,'Resolution Mode','Megapixel + Aspect Ratio');
        assert.deepEqual(n.serialize(),saved);
    }
}
// Legacy states remain literal, including ignored/stale aspect and manual fields.
for(const size_source of ['First Image','Legacy Aspect','Manual']) for(const aspect of aspects.slice(0,2).concat(['Landscape 16:9','Portrait 9:16'])) {
    const inputs={size_source,aspect,preset:presets[1],custom_mp:0.37,width:704,height:1024};
    const n=make(inputs);
    assert.deepEqual(n.serialize(),n.original);
    rows.push({inputs,geometry:null,preview:structuredClone(n.__h3ContinuumResolutionPreview())});
    n.configure(n.original);
    assert.deepEqual(n.serialize(),n.original);
}
const unknown=make({size_source:'Megapixel + Aspect Ratio'},null,true);
assert.equal(unknown.__h3ContinuumResolutionPreview().pending,true);
assert.equal('width' in unknown.__h3ContinuumResolutionPreview(),false);
assert.deepEqual(unknown.serialize(),unknown.original);
const editAspect=make({size_source:'First Image',aspect:'Portrait 9:16'},[1200,800]);
assert.equal(widget(editAspect,'Aspect Ratio').value,'Auto / First Image');
edit(editAspect,'Aspect Ratio','3:2');
assert.equal(widget(editAspect,'aspect').value,'3:2');
assert.equal(widget(editAspect,'size_source').value,'Megapixel + Aspect Ratio');
assert.equal(widget(editAspect,'width').value,960);
// Historical V3.8 migration and preview writes remain deliberately unchanged.
const old=make({aspect:'Landscape 16:9',size_source:'Legacy Aspect'},null,false,'H3ContinuumSamplerV38');
assert.equal(widget(old,'size_source').value,'Manual');
assert.equal(widget(old,'width').value,736);
assert.equal(widget(old,'height').value,416);
assert.equal(widget(old,'Resolution Mode'),undefined);
const oldAuto=make({size_source:'First Image'},[1200,800],false,'H3ContinuumSamplerV38');
assert.equal(widget(oldAuto,'width').value,672);
assert.equal(widget(oldAuto,'height').value,448);
console.log(JSON.stringify(rows));
