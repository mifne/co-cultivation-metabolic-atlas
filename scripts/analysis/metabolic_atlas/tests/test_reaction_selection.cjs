const fs=require('fs'),assert=require('assert');
const cytoscape=require('../web/vendor/cytoscape.min.js');
const src=fs.readFileSync(__dirname+'/../web/js/flow.js','utf8');
const cy=cytoscape({headless:true,styleEnabled:true,elements:[...['glu','akg','cit','nad'].map(id=>({data:{id,kind:'fm'}})),...['GLUDxi','CS','OTHER'].map(id=>({data:{id,kind:'fr',reaction:id}})),... [['glu','GLUDxi'],['GLUDxi','akg'],['akg','CS'],['CS','cit'],['nad','CS'],['glu','OTHER']].map(([source,target],i)=>({data:{id:'e'+i,source,target},classes:i===4?'reactionSide peripheral':'reactionBackbone peripheral'}))]});
const st={keep:new Set(['GLUDxi']),majorShown:new Map([['tca',{pathway:{reactions:['CS']}}]])};
const code=src.slice(src.indexOf('function applyReactionSelection()'),src.indexOf('/* Additional medium roots'));
const apply=new Function('flowMode','flowState','mapCy','appendFlowStyle',code+';return applyReactionSelection')(true,st,cy,(cy,r)=>cy.style().append(r));
apply();for(const id of ['e0','e1','e2','e3']){assert(cy.$id(id).hasClass('selectedReaction'));assert.equal(cy.$id(id).style('opacity'),'1')};assert.equal(cy.$id('e4').style('opacity'),'0.8');assert(!cy.$id('e5').hasClass('selectedReaction'));
st.majorShown.clear();apply();assert(!cy.$id('CS').hasClass('selectedReaction'));assert(!cy.$id('e3').hasClass('selectedReaction'));cy.$id('GLUDxi').addClass('userOmitted');apply();assert(!cy.$id('GLUDxi').hasClass('selectedReaction'));cy.destroy();console.log('PASS selected reaction both sides, pathway expansion, side branch contrast, collapse and omission');
