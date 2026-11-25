"""Build a self-contained interactive HTML dashboard from the demo timelines."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from swn.config import ACTIVITY_LABEL_CN, ARTIFACT_DIR, RESULT_DIR, TIMELINE_DIR  # noqa: E402

TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sense When Needed - 回放看板</title>
<style>
  :root{
    --bg:#f4f6fa; --panel:#ffffff; --ink:#1c2733; --muted:#5b6b7b;
    --line:#dbe2ea; --accent:#2f7ed8; --accent-soft:#e8f1fb;
    --warn:#d1495b; --ok:#3f8f5f; --gps:#8a6d3b;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
       font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif;font-size:14px}
  .wrap{max-width:1080px;margin:0 auto;padding:20px 18px 40px}
  h1{font-size:20px;margin:2px 0 2px}
  .sub{color:var(--muted);font-size:12.5px;margin-bottom:14px}
  .row{display:flex;gap:14px;flex-wrap:wrap}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
  .metrics{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin-bottom:14px}
  .metric .v{font-size:19px;font-weight:650}
  .metric .k{font-size:11px;color:var(--muted);margin-top:2px}
  .metric.alert .v{color:var(--warn)}
  .seg{display:inline-flex;border:1px solid var(--line);border-radius:10px;overflow:hidden;margin-bottom:12px}
  .seg button{border:0;background:transparent;padding:7px 14px;cursor:pointer;font-size:13px;color:var(--muted)}
  .seg button.on{background:var(--accent);color:#fff}
  canvas{width:100%;display:block;border-radius:8px}
  .grid2{display:grid;grid-template-columns:1fr 320px;gap:14px}
  @media (max-width:900px){.grid2{grid-template-columns:1fr}.metrics{grid-template-columns:repeat(3,1fr)}}
  .controls{display:flex;align-items:center;gap:10px;margin:10px 0 6px;flex-wrap:wrap}
  .controls button{border:1px solid var(--line);background:#fff;border-radius:8px;padding:5px 12px;cursor:pointer;font-size:13px}
  .controls button.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
  .controls input[type=range]{flex:1}
  .t{font-variant-numeric:tabular-nums;color:var(--muted);font-size:12px;min-width:88px}
  .legend{display:flex;gap:12px;flex-wrap:wrap;font-size:11.5px;color:var(--muted);margin-top:6px}
  .legend i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:4px;vertical-align:-1px}
  .statecard h3{margin:0 0 8px;font-size:13px;color:var(--muted);font-weight:600}
  .big{font-size:24px;font-weight:700}
  .kv{display:flex;justify-content:space-between;font-size:12.5px;padding:3px 0;border-bottom:1px dashed var(--line)}
  .kv:last-child{border-bottom:0}
  .feed{max-height:260px;overflow:auto;font-size:12px}
  .feed .item{border-left:3px solid var(--accent);padding:6px 8px;margin:8px 0;background:var(--accent-soft);border-radius:0 8px 8px 0}
  .feed .item .meta{color:var(--muted);font-size:10.5px;margin-bottom:3px}
  .footer{color:var(--muted);font-size:11px;margin-top:16px;line-height:1.6}
  .pill{display:inline-block;padding:1px 8px;border-radius:999px;font-size:11px;background:var(--accent-soft);color:var(--accent)}
  .pill.gps{background:#f6ecd9;color:var(--gps)}
</style>
</head>
<body>
<div class="wrap">
  <h1>Sense When Needed · 事件驱动自适应多传感活动理解</h1>
  <div class="sub">真实 WISDM 腕部数据回放 · 会话 <b id="sess">-</b> · 三种采样策略同一数据、同一模型，仅"何时开机"不同</div>

  <div class="seg" id="tabs"></div>

  <div class="metrics" id="metrics"></div>

  <div class="grid2">
    <div class="card">
      <div class="controls">
        <button class="primary" id="play">▶ 播放</button>
        <span class="t" id="clock">00:00 / 00:00</span>
        <input type="range" id="scrub" min="0" max="100" value="0">
        <select id="speed"><option>1×</option><option selected>4×</option><option>8×</option><option>16×</option></select>
      </div>
      <canvas id="tl" height="360"></canvas>
      <div class="legend" id="legend"></div>
    </div>

    <div>
      <div class="card statecard" style="margin-bottom:14px">
        <h3>当前状态</h3>
        <div class="big" id="curAct">—</div>
        <div style="color:var(--muted);font-size:12px;margin-bottom:8px" id="curState">—</div>
        <div class="kv"><span>真实活动</span><b id="curTrue">—</b></div>
        <div class="kv"><span>系统估计</span><b id="curPred">—</b></div>
        <div class="kv"><span>置信度</span><b id="curConf">—</b></div>
        <div class="kv"><span>传感模式</span><b id="curMode">—</b></div>
        <div class="kv"><span>P(change)</span><b id="curProb">—</b></div>
        <div style="margin-top:8px" id="badges"></div>
      </div>
      <div class="card statecard">
        <h3>LLM 情境摘要（仅重要转换时调用）</h3>
        <div class="feed" id="feed"><div style="color:var(--muted)">尚无调用…</div></div>
      </div>
    </div>
  </div>

  <div class="footer" id="foot"></div>
</div>

<script>
const DATA = __DATA__;

const CN = {walking:"行走", jogging:"慢跑", stairs:"上下楼", sitting:"静坐", standing:"站立", unknown:"未知"};
const COL = {walking:"#4e79a7", jogging:"#f28e2b", stairs:"#59a14f", sitting:"#e15759", standing:"#b07aa1", unknown:"#c8cfd6"};
const ORDER = ["walking","jogging","stairs","sitting","standing"];
const STATE_CN = {idle:"低功耗监测（1s 突发 / 6s）", active:"主动感知（20 Hz）", sampling:"周期采样", sleep:"休眠", context:"LLM 情境理解"};

const STRATS = [
  {id:"always_on",   name:"Always-on（20 Hz 常开）"},
  {id:"periodic",    name:"Periodic（3s / 10s）"},
  {id:"event_driven",name:"Event-driven（本文方法）"},
];

let cur = "event_driven";
let t = 0, playing = false, lastTs = null;

function fmt(s){s=Math.max(0,s|0);return String((s/60)|0).padStart(2,"0")+":"+String(s%60).padStart(2,"0");}

function buildTabs(){
  const el=document.getElementById("tabs"); el.innerHTML="";
  STRATS.forEach(st=>{
    const b=document.createElement("button");
    b.textContent=st.name; if(st.id===cur) b.classList.add("on");
    b.onclick=()=>{cur=st.id; t=0; buildTabs(); buildMetrics(); draw(); update(); renderFeed(true);};
    el.appendChild(b);
  });
}

function buildMetrics(){
  const s=DATA.strategies[cur], c=s.counters, el=document.getElementById("metrics");
  const items=[
    ["跟踪准确率",(s.accuracy*100).toFixed(1)+"%",""],
    ["Macro F1",s.macro_f1.toFixed(3),""],
    ["IMU 占空比",(c.duty_cycle*100).toFixed(1)+"%", c.duty_cycle<0.35?"":"alert"],
    ["能耗代理",c.energy_j_per_hour.toFixed(2)+" J/h",""],
    ["推理次数/h",Math.round(c.inferences_per_hour),""],
    ["LLM 调用/h",Math.round(c.llm_calls_per_hour),""],
  ];
  el.innerHTML=items.map(([k,v,cls])=>`<div class="card metric ${cls}"><div class="v">${v}</div><div class="k">${k}</div></div>`).join("");
  document.getElementById("foot").innerHTML =
    `能耗代理 = 功率模型(20Hz IMU 1.0mW · 待机 0.02mW · GPS 30mW) × 实际开机时间，为资源消耗代理指标而非实测电池续航；
     GPS 仅在"由静入动"时开启（本次会话 ${DATA.strategies[cur].counters.gps_on_s.toFixed(0)} s）；
     LLM 后端：${(s.llm&&s.llm.backend)||"-"}（离线模板 / 可切换 OpenAI 兼容 API）；
     数据：WISDM 腕部 IMU，按受试者划分训练/测试。`;
}

function run(){ return DATA.strategies[cur]; }
function T(){ return run().timeline; }

function draw(){
  const tl=T(), cv=document.getElementById("tl"), dpr=window.devicePixelRatio||1;
  const W=cv.clientWidth, H=360;
  cv.width=W*dpr; cv.height=H*dpr; cv.style.height=H+"px";
  const g=cv.getContext("2d"); g.scale(dpr,dpr);
  g.clearRect(0,0,W,H);
  const dur=run().duration;
  const x=t0=>t0/dur*(W-70)+70;
  const lanes=[
    {y:10,h:52,title:"真实活动",key:"true_name"},
    {y:72,h:52,title:"系统估计",key:"pred_name"},
    {y:134,h:46,title:"传感状态",key:null},
    {y:190,h:74,title:"事件概率",key:null},
  ];
  // lane backgrounds
  lanes.forEach(L=>{ g.fillStyle="#fbfcfe"; g.strokeStyle="#dbe2ea";
    g.beginPath(); g.rect(70,L.y,W-70,L.h); g.fill(); g.stroke();
    g.fillStyle="#5b6b7b"; g.font="11px Segoe UI"; g.textBaseline="middle";
    g.fillText(L.title,4,L.y+L.h/2); });
  // ribbons
  const seg=(key)=>{ let start=0, curV=tl[0][key];
    for(let i=1;i<tl.length;i++){ if(tl[i][key]!==curV){ paint(key,curV,start,tl[i].t); start=tl[i].t; curV=tl[i][key]; } }
    paint(key,curV,start,run().duration);
    function paint(k,v,a,b){ if(k==="true_name"||k==="pred_name"){
      g.fillStyle=COL[v]||"#ccc"; g.globalAlpha=(k==="pred_name")?0.92:1;
      g.fillRect(x(a),lanes[0].y+(k==="pred_name"?62:0),Math.max(1,x(b)-x(a)),52); g.globalAlpha=1;
    }};
  };
  seg("true_name"); seg("pred_name");
  // state lane
  let a=0,st=tl[0].state;
  for(let i=1;i<tl.length;i++){ if(tl[i].state!==st){ stateSeg(st,a,tl[i].t); a=tl[i].t; st=tl[i].state; } }
  stateSeg(st,a,run().duration);
  function stateSeg(s0,a0,b0){
    const c = s0==="active" ? "#2f7ed8" : (s0==="context" ? "#3f8f5f" : (s0==="sampling" ? "#f0a45c" : "#cfe0f3"));
    g.fillStyle=c; g.fillRect(x(a0),lanes[2].y,Math.max(1,x(b0)-x(a0)),lanes[2].h);
  }
  // transitions + gps + llm markers
  run().transitions.forEach(tr=>{ g.strokeStyle="rgba(0,0,0,0.35)"; g.setLineDash([3,3]);
    g.beginPath(); g.moveTo(x(tr.t),lanes[0].y); g.lineTo(x(tr.t),lanes[2].y+lanes[2].h); g.stroke(); g.setLineDash([]); });
  tl.forEach(r=>{ if(r.gps){ g.fillStyle="#c9a227"; g.fillRect(x(r.t),lanes[2].y-4,Math.max(1,(W-70)/dur),4);} });
  run().llm_calls.forEach(c=>{ g.fillStyle="#3f8f5f"; g.beginPath(); g.arc(x(c.t),lanes[2].y+lanes[2].h+7,4,0,7); g.fill(); });
  run().triggers.forEach(c=>{ g.strokeStyle="#d1495b"; g.beginPath(); g.moveTo(x(c.t),lanes[3].y); g.lineTo(x(c.t),lanes[3].y+lanes[3].h); g.stroke(); });
  // probability
  g.strokeStyle="#2f7ed8"; g.lineWidth=1.2; g.beginPath(); let started=false;
  tl.forEach(r=>{ if(r.event_prob==null) return; const px=x(r.t),py=lanes[3].y+lanes[3].h-(r.event_prob)*lanes[3].h;
    if(!started){g.moveTo(px,py);started=true;} else g.lineTo(px,py); });
  g.stroke();
  g.strokeStyle="#d1495b"; g.setLineDash([4,3]); g.beginPath();
  g.moveTo(70,lanes[3].y+lanes[3].h*0.5); g.lineTo(W,lanes[3].y+lanes[3].h*0.5); g.stroke(); g.setLineDash([]);
  g.fillStyle="#5b6b7b"; g.font="10px Segoe UI"; g.fillText("阈值 0.5",W-58,lanes[3].y+lanes[3].h*0.5-6);
  // time axis
  g.fillStyle="#5b6b7b"; g.font="10px Segoe UI";
  for(let s=0;s<=dur;s+=60){ g.fillText(fmt(s),x(s)-16,lanes[3].y+lanes[3].h+34);
    g.strokeStyle="#e6ebf1"; g.beginPath(); g.moveTo(x(s),lanes[0].y); g.lineTo(x(s),lanes[3].y+lanes[3].h); g.stroke(); }
  // playhead
  const px=x(Math.min(t,dur));
  g.strokeStyle="#d1495b"; g.lineWidth=1.6; g.beginPath(); g.moveTo(px,lanes[0].y-4); g.lineTo(px,lanes[3].y+lanes[3].h+4); g.stroke();
}

function at(tt){ const tl=T(); let lo=0,hi=tl.length-1;
  while(lo<hi){const m=(lo+hi)>>1; if(tl[m].t<tt) lo=m+1; else hi=m;} return tl[Math.min(lo,tl.length-1)]; }

function update(){
  const r=at(t), s=run();
  document.getElementById("curTrue").textContent=CN[r.true_name]||r.true_name;
  document.getElementById("curPred").textContent=CN[r.pred_name]||r.pred_name;
  document.getElementById("curPred").style.color = r.pred===r.true ? "var(--ok)" : "var(--warn)";
  document.getElementById("curAct").textContent=CN[r.pred_name]||"—";
  document.getElementById("curState").textContent=STATE_CN[r.state]||r.state;
  document.getElementById("curConf").textContent=r.conf.toFixed(2);
  document.getElementById("curMode").textContent=r.imu_on?"20 Hz 持续":"1 s 突发 / 6 s";
  document.getElementById("curProb").textContent=r.event_prob==null?"—":r.event_prob.toFixed(2);
  const b=[];
  if(r.gps) b.push('<span class="pill gps">GPS 定位中</span>');
  if(r.state==="active") b.push('<span class="pill">高保真确认中</span>');
  document.getElementById("badges").innerHTML=b.join(" ");
  document.getElementById("clock").textContent=fmt(t)+" / "+fmt(run().duration);
  document.getElementById("scrub").value=Math.round(t/run().duration*100);
  renderFeed();
}

let fedUpTo=0;
function renderFeed(reset){
  const el=document.getElementById("feed");
  if(reset){ el.innerHTML=""; fedUpTo=0; }
  const calls=run().llm_calls.filter(c=>c.t<=t && c.t>fedUpTo-1e-9 && c.t>fedUpTo-1);
  const all=run().llm_calls.filter(c=>c.t<=t);
  if(all.length===0){ el.innerHTML='<div style="color:var(--muted)">尚无调用…</div>'; fedUpTo=t; return; }
  el.innerHTML=all.slice(-6).reverse().map(c=>`
    <div class="item"><div class="meta">t=${c.t.toFixed(0)}s · ${CN[c.current_activity]||c.current_activity}
      （置信度 ${c.confidence}）· ${c.backend}</div>${c.text}</div>`).join("");
  fedUpTo=Math.max(fedUpTo,...all.map(c=>c.t));
}

function loop(ts){
  if(playing){
    if(lastTs!=null){ const sp=parseFloat(document.getElementById("speed").value)||1;
      t=Math.min(run().duration, t+(ts-lastTs)/1000*sp); update(); draw(); }
    lastTs=ts;
    if(t>=run().duration){ playing=false; document.getElementById("play").textContent="▶ 播放"; }
  } else lastTs=ts;
  requestAnimationFrame(loop);
}

document.getElementById("play").onclick=()=>{ playing=!playing;
  if(t>=run().duration) t=0;
  document.getElementById("play").textContent=playing?"⏸ 暂停":"▶ 播放"; };
document.getElementById("scrub").oninput=e=>{ t=e.target.value/100*run().duration; update(); draw(); };
window.addEventListener("resize",draw);

const lg=document.getElementById("legend");
lg.innerHTML=ORDER.map(a=>`<span><i style="background:${COL[a]}"></i>${CN[a]}</span>`).join("")
 +`<span><i style="background:#cfe0f3"></i>低功耗</span><span><i style="background:#2f7ed8"></i>20 Hz</span>`
 +`<span><i style="background:#c9a227"></i>GPS</span><span><i style="background:#3f8f5f"></i>LLM 调用</span>`;

document.getElementById("sess").textContent=DATA.strategies.event_driven.session_id+"（受试者 "+DATA.strategies.event_driven.subject+"）";
buildTabs(); buildMetrics(); update(); draw(); requestAnimationFrame(loop);
</script>
</body>
</html>
"""

def main() -> None:
    comp = json.loads((RESULT_DIR / "experiments.json").read_text(encoding="utf-8"))
    demo_path = TIMELINE_DIR / f"demo_{comp['demo_session_id']}.json"
    demo = json.loads(demo_path.read_text(encoding="utf-8"))
    payload = {
        "demo_session_id": comp["demo_session_id"],
        "strategies": {
            k: {
                "accuracy": comp["strategies"][k]["accuracy"],
                "macro_f1": comp["strategies"][k]["macro_f1"],
                "llm": comp["strategies"][k].get("llm"),
                "session_id": v["session_id"],
                "subject": v["subject"],
                "duration": v["duration"],
                "timeline": v["timeline"],
                "triggers": v["triggers"],
                "transitions": v["transitions"],
                "llm_calls": v["llm_calls"],
                "counters": {
                    **v["counters"],
                    "energy_j_per_hour": v["counters"]["energy_mj"] / v["duration"] * 3.6,
                    "llm_calls_per_hour": v["counters"]["llm_calls"] / v["duration"] * 3600.0,
                    "inferences_per_hour": v["counters"]["inferences"] / v["duration"] * 3600.0,
                    "samples_per_hour": v["counters"]["samples"] / v["duration"] * 3600.0,
                },
            }
            for k, v in demo.items()
        },
        "accuracy": {k: comp["strategies"][k]["accuracy"] for k in demo},
        "macro_f1": {k: comp["strategies"][k]["macro_f1"] for k in demo},
    }
    html = TEMPLATE.replace("__DATA__", json.dumps(payload, ensure_ascii=False))
    out = ARTIFACT_DIR / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
