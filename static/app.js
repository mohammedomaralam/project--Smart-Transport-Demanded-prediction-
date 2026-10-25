document.addEventListener("DOMContentLoaded", () => {
  const date = document.querySelector('input[type="date"]');
  if (date && !date.value) {
    date.value = new Date().toISOString().slice(0, 10);
    date.min = new Date().toISOString().slice(0, 10);
  }
  document.querySelectorAll("form").forEach(form => {
    form.addEventListener("submit", () => {
      const btn = form.querySelector("button[type=submit], button");
      if (btn && !btn.classList.contains("no-loading")) {
        btn.dataset.original = btn.innerHTML;
        btn.innerHTML = "Please wait...";
      }
    });
  });
  if (document.getElementById("routeChart")) drawCharts();
  document.querySelectorAll(".password-toggle").forEach(toggle => {
    toggle.addEventListener("click", () => {
      const input = toggle.parentElement.querySelector("input");
      const reveal = input.type === "password";
      input.type = reveal ? "text" : "password";
      toggle.textContent = reveal ? "Hide" : "Show";
      toggle.setAttribute("aria-label", reveal ? "Hide password" : "Show password");
    });
  });
  const registerForm = document.querySelector('form input[name="confirm_password"]')?.form;
  if (registerForm) registerForm.addEventListener("submit", event => {
    const password = registerForm.querySelector('input[name="password"]');
    const confirm = registerForm.querySelector('input[name="confirm_password"]');
    if (password && confirm && password.value !== confirm.value) {
      event.preventDefault();
      confirm.setCustomValidity("Passwords do not match.");
      confirm.reportValidity();
    }
  });
  document.querySelectorAll('input[name="confirm_password"]').forEach(input => {
    input.addEventListener("input", () => input.setCustomValidity(""));
  });
});
function setupCanvas(id) {
  const c = document.getElementById(id);
  if (!c) return null;
  const ctx = c.getContext("2d");
  const rect = c.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  c.width = Math.max(280, rect.width) * dpr;
  c.height = 205 * dpr;
  ctx.scale(dpr, dpr);
  return {ctx, w: Math.max(280, rect.width), h: 205};
}
function drawBars(id, labels, values) {
  const s = setupCanvas(id); if (!s) return;
  const {ctx,w,h}=s, left=32, bottom=34, top=18, right=10, max=Math.max(...values)*1.15;
  ctx.clearRect(0,0,w,h); ctx.font="10px Segoe UI, Arial"; ctx.strokeStyle="#e7edf5";
  for(let k=0;k<=4;k++){const y=top+(h-top-bottom)*k/4;ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(w-right,y);ctx.stroke();}
  const cw=w-left-right, gap=10, bw=(cw-gap*(labels.length-1))/labels.length;
  values.forEach((v,i)=>{const bh=(v/max)*(h-top-bottom),x=left+i*(bw+gap),y=h-bottom-bh;ctx.fillStyle="#347fd0";ctx.fillRect(x,y,bw,bh);ctx.fillStyle="#46566b";ctx.textAlign="center";ctx.fillText(String(v),x+bw/2,y-5);ctx.fillText(labels[i],x+bw/2,h-12);});
}
function drawLine(id, labels, values) {
  const s=setupCanvas(id);if(!s)return;
  const {ctx,w,h}=s,left=32,bottom=34,top=18,right=12,max=Math.max(...values)*1.15,cw=w-left-right,ch=h-top-bottom;
  ctx.clearRect(0,0,w,h);ctx.strokeStyle="#e7edf5";
  for(let k=0;k<=4;k++){const y=top+ch*k/4;ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(w-right,y);ctx.stroke();}
  ctx.strokeStyle="#347fd0";ctx.lineWidth=2;ctx.beginPath();
  values.forEach((v,i)=>{const x=left+i*cw/(values.length-1),y=h-bottom-v/max*ch;i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();
  values.forEach((v,i)=>{const x=left+i*cw/(values.length-1),y=h-bottom-v/max*ch;ctx.beginPath();ctx.arc(x,y,3,0,Math.PI*2);ctx.fillStyle="#347fd0";ctx.fill();ctx.fillStyle="#46566b";ctx.font="9px Segoe UI, Arial";ctx.textAlign="center";ctx.fillText(labels[i],x,h-12);});
}
function drawDonut(id, values, labels) {
  const s=setupCanvas(id);if(!s)return;const {ctx,w,h}=s,cx=w/2,cy=90,r=62,total=values.reduce((a,b)=>a+b,0);
  const colors=["#65a97b","#e7b64b","#e58c48","#d85f5f"];let start=-Math.PI/2;ctx.clearRect(0,0,w,h);
  values.forEach((v,i)=>{const a=v/total*Math.PI*2;ctx.beginPath();ctx.moveTo(cx,cy);ctx.arc(cx,cy,r,start,start+a);ctx.closePath();ctx.fillStyle=colors[i];ctx.fill();start+=a;});
  ctx.beginPath();ctx.arc(cx,cy,r*.58,0,Math.PI*2);ctx.fillStyle="#fff";ctx.fill();
  ctx.fillStyle="#334155";ctx.font="bold 17px Segoe UI";ctx.textAlign="center";ctx.fillText("10K",cx,cy+6);
  const legendY=190;labels.forEach((l,i)=>{const x=12+i*(w/4);ctx.fillStyle=colors[i];ctx.fillRect(x,legendY-7,7,7);ctx.fillStyle="#64748b";ctx.font="9px Segoe UI";ctx.textAlign="left";ctx.fillText(l,x+10,legendY);});
}
function drawCharts(){
  const d=window.ANALYTICS; if(!d) return;
  drawBars("routeChart",d.route.labels,d.route.values);
  drawLine("timeChart",d.hour.labels,d.hour.values);
  drawBars("weatherChart",d.weather.labels,d.weather.values);
  drawBars("crowdChart",d.daytype.labels,d.daytype.values);
}
window.addEventListener("resize",()=>{if(document.getElementById("routeChart"))drawCharts();});
