const fs=require('fs'),vm=require('vm'),path=require('path'),assert=require('assert');
const root=path.resolve(__dirname,'../..');const cytoscape=require(path.join(root,'outputs/metabolic_map_20260922/vendor/cytoscape.min.js'));
const src=fs.readFileSync(path.join(__dirname,'metabolic_map_flow.js'),'utf8');
const cy=cytoscape({headless:true,elements:[{data:{id:'m',kind:'fm'},position:{x:20,y:30}},{data:{id:'fold_m',kind:'fold',target:'m'},position:{x:0,y:0}}],layout:{name:'preset'}});
new Function('mapCy',src.slice(src.indexOf('function bindFoldAnchors()'),src.indexOf('const anchorRenderBase='))+';bindFoldAnchors();')(cy);
function check(){const p=cy.$id('m').position(),q=cy.$id('fold_m').position();assert.equal(q.x,p.x);assert.equal(q.y,p.y+48);assert.equal(cy.$id('fold_m').grabbable(),false)}
check();cy.$id('m').position({x:240,y:-80});check();cy.batch(()=>cy.$id('m').position({x:-140,y:500}));check();cy.$id('fold_m').remove();cy.add({data:{id:'fold_m',kind:'fold',target:'m'}});check();cy.pan({x:30,y:50});cy.zoom(1.3);check();console.log('PASS: initial anchoring, position change, batched layout, badge recreation, pan/zoom, non-draggable badge');cy.destroy();
