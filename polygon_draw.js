export default function ({ parentElement, data, setStateValue }) {
  const canvas = parentElement.querySelector("canvas");
  const undo = parentElement.querySelector("[data-undo]");
  const clear = parentElement.querySelector("[data-clear]");
  const apply = parentElement.querySelector("[data-apply]");
  const status = parentElement.querySelector("output");
  const { nx, ny, vertices = [] } = data;
  const ctx = canvas.getContext("2d");
  const width = canvas.width = 800;
  const height = canvas.height = Math.round(width * ny / nx);
  canvas.style.maxWidth = Math.min(width, 300 * nx / ny) + "px";
  const applied = JSON.stringify(vertices);
  let points = JSON.parse(canvas.dataset.draft || applied);

  function draw() {
    ctx.clearRect(0, 0, width, height);
    ctx.strokeStyle = "#dce5eb";
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 1; i < nx; i++) {
      ctx.moveTo(i * width / nx, 0);
      ctx.lineTo(i * width / nx, height);
    }
    for (let j = 1; j < ny; j++) {
      ctx.moveTo(0, j * height / ny);
      ctx.lineTo(width, j * height / ny);
    }
    ctx.stroke();
    ctx.beginPath();
    points.forEach(([x, y], index) => {
      if (index === 0) ctx.moveTo(x * width, (1-y) * height);
      else ctx.lineTo(x * width, (1-y) * height);
    });
    if (points.length >= 3) {
      ctx.closePath();
      ctx.fillStyle = "#057f9436";
      ctx.fill("evenodd");
    }
    ctx.lineWidth = 3;
    ctx.strokeStyle = "#057f94";
    ctx.stroke();
    ctx.fillStyle = "#057f94";
    for (const [x, y] of points) {
      ctx.beginPath();
      ctx.arc(x * width, (1-y) * height, 4, 0, 2 * Math.PI);
      ctx.fill();
    }
    const modified = JSON.stringify(points) !== applied;
    status.textContent = `${points.length} 个顶点${modified ? " · 待应用" : ""}`;
    undo.disabled = clear.disabled = points.length === 0;
    apply.disabled = !modified || (points.length > 0 && points.length < 3);
  }

  function update(next) {
    points = next;
    canvas.dataset.draft = JSON.stringify(next);
    draw();
  }

  canvas.onclick = (event) => {
    const rect = canvas.getBoundingClientRect();
    const x = Math.round(nx * (event.clientX-rect.left)/rect.width) / nx;
    const y = Math.round(ny * (rect.bottom-event.clientY)/rect.height) / ny;
    const last = points.at(-1);
    if (!last || last[0] !== x || last[1] !== y) update([...points, [x, y]]);
  };
  undo.onclick = () => update(points.slice(0, -1));
  clear.onclick = () => update([]);
  apply.onclick = () => {
    apply.disabled = true;
    setStateValue("vertices", points);
  };
  draw();
}
