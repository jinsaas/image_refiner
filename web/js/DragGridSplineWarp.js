import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "IRL.DragGridSplineWarp.DragWidget",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "IRL_DragGridSplineWarp") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);

                // 1. Hide the P1–P8 input fields generated in Python from the UI
                function parsePoints(str) {
                    const regex = /(P\d+)\{([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\}/g;
                    const points = [];
                    let match;
                    while ((match = regex.exec(str)) !== null) {
                        points.push({ label: match[1], x: parseFloat(match[2]), y: parseFloat(match[3]) });
                    }
                    return points;
                }

                function serializePoints(points) {
                    return points.map(p => `${p.label}{${p.x.toFixed(3)},${p.y.toFixed(3)}}`).join(", ");
                }

                const node = this;
				
                node.dragAreaSize = 180; // Canvas size (pixels)
                node.selectedPoint = null;

                node.pointKeys = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"];
				// Parse the widget's string value ("x, y") to get the current coordinates
                node.getControlPoints = function() {
                    const w = node.widgets.find(w => w.name === "p_point");
                    if (!w || typeof w.value !== "string") return [];
                    const parsed = parsePoints(w.value);
                    // Do not simply return the array; return the coordinate array sorted in the order P1 to P8
                    return node.pointKeys.map(key => {
                        const found = parsed.find(p => p.label === key);
                        return found ? {x: found.x, y: found.y} : {x:0.0, y:0.0};
                    });
                };


                // When dragging a point, update the value of the actual Python input widget in the "x, y" format
                node.updateControlPoint = function(index, x, y) {
                    const w = node.widgets.find(w => w.name === "p_point");
                    if (!w) return;
                    const points = parsePoints(w.value);
                    const targetLabel = node.pointKeys[index];
                    const target = points.find(p => p.label === targetLabel);
                    if (target) {
                        target.x = parseFloat(x.toFixed(3));
                        target.y = parseFloat(y.toFixed(3));
                    }
                    w.value = serializePoints(points);
                    if (w.callback) w.callback(w.value);
					if (app.graph) {
                        app.graph.change();
                    }
                };

                // Create a custom widget (LiteGraph method)
                const customWidget = {
                    type: "grid_canvas",
                    name: "grid_interactive_canvas",
                    label: "Grid Point Controller",
                    value: "",
                    draw: function(ctx, widgetNode, width, y) {
                        if (widgetNode.flags.collapsed) return;

                        this.last_y = y + 20;

                        ctx.save();
                        const startX = (width - widgetNode.dragAreaSize) / 2;
                        const startY = y + 20;
                        
                        // 1. Background box rendering
                        ctx.fillStyle = "#1a1a1a";
                        ctx.fillRect(startX, startY, widgetNode.dragAreaSize, widgetNode.dragAreaSize);
                        ctx.strokeStyle = "#444";
                        ctx.strokeRect(startX, startY, widgetNode.dragAreaSize, widgetNode.dragAreaSize);

                        // 2. Cross Guidelines
                        ctx.strokeStyle = "#333";
                        ctx.lineWidth = 1;
                        ctx.beginPath();
                        ctx.moveTo(startX + widgetNode.dragAreaSize / 2, startY);
                        ctx.lineTo(startX + widgetNode.dragAreaSize / 2, startY + widgetNode.dragAreaSize);
                        ctx.moveTo(startX, startY + widgetNode.dragAreaSize / 2);
                        ctx.lineTo(startX + widgetNode.dragAreaSize, startY + widgetNode.dragAreaSize / 2);
                        ctx.stroke();

                        // 3. CC (Center point - red marker)
                        const points = widgetNode.getControlPoints();
                        const ccX_val = (points[0].x + points[2].x + points[5].x + points[7].x) / 4.0;
                        const ccY_val = (points[0].y + points[2].y + points[5].y + points[7].y) / 4.0;
                        const ccX = startX + ccX_val * widgetNode.dragAreaSize;
                        const ccY = startY + ccY_val * widgetNode.dragAreaSize;

                        ctx.fillStyle = "#ff4d4d";
                        ctx.beginPath();
                        ctx.arc(ccX, ccY, 4, 0, Math.PI * 2);
                        ctx.fill();
                        ctx.fillStyle = "#ff8888";
                        ctx.font = "9px sans-serif";
                        ctx.fillText("CC", ccX + 6, ccY + 3);

                        // 4. Rendering of points P1 to P8
                        points.forEach((p, idx) => {
                            const px = startX + p.x * widgetNode.dragAreaSize;
                            const py = startY + p.y * widgetNode.dragAreaSize;
                            
                            ctx.fillStyle = (widgetNode.selectedPoint === idx) ? "#2ecc71" : "#3498db";
                            ctx.beginPath();
                            ctx.arc(px, py, 5, 0, Math.PI * 2);
                            ctx.fill();

                            ctx.fillStyle = "#bbb";
                            ctx.fillText(`P${idx+1}`, px + 7, py + 3);
                        });

                        ctx.restore();
                    },
                    computeSize: function(width) {
                        return [width, node.dragAreaSize + 35];
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
                                if (Math.hypot(localX - px, localY - py) < 12) {
                                    widgetNode.selectedPoint = i;
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

                                const w = widgetNode.widgets?.find(w => w.name === "p_point");
                                if (w) {
                                    if (w.callback) w.callback(w.value);
                                }
                                if (app.graph) {
                                    app.graph.change();
                                }
                                widgetNode.setDirtyCanvas(true, true);
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