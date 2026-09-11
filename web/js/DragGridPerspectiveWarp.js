import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "IRL.DragPerspectiveWarp.DragWidget",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "IRL_DragGridGuidancePerspectiveWarp") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);

                function serializePoint(label, x, y) {
                    return `${label}{${x.toFixed(3)},${y.toFixed(3)}}`;
                }

                const node = this;
                node.dragAreaSize = 180; // Canvas size (pixels)
                node.selectedPoint = null;

                // An array of P1 to P4 key names mapped to Python widget names
                node.pointKeys = ["dst_p1", "dst_p2", "dst_p3", "dst_p4"];

                // 각 위젯 문자열에서 중괄호 안의 좌표를 안전하게 파싱
                function parsePointStr(str) {
                    const match = String(str).match(/\{([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\}/);
                    if (match) {
                        return { x: parseFloat(match[1]), y: parseFloat(match[2]) };
                    }
                    return { x: 0.0, y: 0.0 };
                }

                // Retrieves the coordinates of the current points from the widget value
                node.getControlPoints = function() {
                    return node.pointKeys.map(k => {
                        const w = node.widgets.find(w => w.name === k);
                        return w ? parsePointStr(w.value) : { x: 0.0, y: 0.0 };
                    });
                };

                // Refresh the value of the actual Python input widget when dragging a point
                node.updateControlPoint = function(index, x, y) {
                    const key = node.pointKeys[index];
                    const w = node.widgets.find(w => w.name === key);
                    if (!w) return;

                    const cleanX = parseFloat(x.toFixed(3));
                    const cleanY = parseFloat(y.toFixed(3));
                    w.value = serializePoint(`P${index + 1}`, cleanX, cleanY);
                    
                    if (w.callback) w.callback(w.value);
                };

                const customWidget = {
                    type: "perspective_canvas",
                    name: "perspective_interactive_canvas",
                    label: "Perspective Point Controller",
                    value: "",
                    draw: function(ctx, widgetNode, width, y) {
                        if (widgetNode.flags.collapsed) return;

                        this.last_y = y + 20;

                        ctx.save();
                        const startX = (width - widgetNode.dragAreaSize) / 2;
                        const startY = y + 20;

                        // Background box[cite: 2]
                        ctx.fillStyle = "#1a1a1a";
                        ctx.fillRect(startX, startY, widgetNode.dragAreaSize, widgetNode.dragAreaSize);
                        ctx.strokeStyle = "#444";
                        ctx.strokeRect(startX, startY, widgetNode.dragAreaSize, widgetNode.dragAreaSize);

                        // Square connecting line (P1 -> P2 -> P4 -> P3 -> P1)
                        const points = widgetNode.getControlPoints();
                        ctx.strokeStyle = "#e67e22";
                        ctx.lineWidth = 1.5;
                        ctx.beginPath();
                        if (points.length === 4) {
                            const order = [0, 1, 3, 2];
                            ctx.moveTo(startX + points[order[0]].x * widgetNode.dragAreaSize, startY + points[order[0]].y * widgetNode.dragAreaSize);
                            for (let i = 1; i < order.length; i++) {
                                ctx.lineTo(startX + points[order[i]].x * widgetNode.dragAreaSize, startY + points[order[i]].y * widgetNode.dragAreaSize);
                            }
                            ctx.closePath();
                        }
                        ctx.stroke();

                        // P1~P4 point rendering
                        points.forEach((p, idx) => {
                            const px = startX + p.x * widgetNode.dragAreaSize;
                            const py = startY + p.y * widgetNode.dragAreaSize;

                            ctx.fillStyle = (widgetNode.selectedPoint === idx) ? "#2ecc71" : "#3498db";
                            ctx.beginPath();
                            ctx.arc(px, py, 6, 0, Math.PI * 2);
                            ctx.fill();

                            ctx.fillStyle = "#bbb";
                            ctx.font = "10px sans-serif";
                            ctx.fillText(`P${idx + 1}`, px + 8, py + 3);
                        });

                        ctx.restore();
                    },
                    computeSize: function(width) {
                        return [width, node.dragAreaSize + 35]; // 스플라인과 동일한 여백[cite: 2]
                    },
                    mouse: function(event, pos, widgetNode) {
                        const startX = (widgetNode.size[0] - widgetNode.dragAreaSize) / 2;
                        const startY = this.last_y || 20;

                        const localX = pos[0] - startX;
                        const localY = pos[1] - startY;

                        if (event.type === "pointerdown" || event.type === "mousedown") {
                            const points = widgetNode.getControlPoints();
                            for (let i = 0; i < points.length; i++) {
                                const px = points[i].x * widgetNode.dragAreaSize;
                                const py = points[i].y * widgetNode.dragAreaSize;
                                if (Math.hypot(localX - px, localY - py) < 15) {
                                    widgetNode.selectedPoint = i;
                                    widgetNode.setDirtyCanvas(true, true);
                                    return true;
                                }
                            }
                            widgetNode.selectedPoint = null;
                        } else if (event.type === "pointermove" || event.type === "mousemove") {
                            if (widgetNode.selectedPoint !== null) {
                                const nx = Math.max(0, Math.min(1, localX / widgetNode.dragAreaSize));
                                const ny = Math.max(0, Math.min(1, localY / widgetNode.dragAreaSize));
                                
                                widgetNode.updateControlPoint(widgetNode.selectedPoint, nx, ny);
                                widgetNode.setDirtyCanvas(true, true);
                                return true;
                            }
                        } else if (event.type === "pointerup" || event.type === "mouseup") {
                            if (widgetNode.selectedPoint !== null) {
                                widgetNode.selectedPoint = null;

                                // 스플라인처럼 마우스를 뗄 때 모든 위젯 콜백 및 그래프 변경 사항 강제 동기화[cite: 2]
                                widgetNode.pointKeys.forEach(k => {
                                    const w = widgetNode.widgets?.find(w => w.name === k);
                                    if (w && w.callback) w.callback(w.value);
                                });
                                if (app.graph) {
                                    app.graph.change();
                                }
                                widgetNode.setDirtyCanvas(true, true);
                                return true;
                            }
                        }
                        return false;
                    }
                };

                this.addCustomWidget(customWidget);
            };
        }
    }
});