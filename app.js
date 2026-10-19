// ============================================================
// YOLO11 SEGMENTATION REALTIME WEB APP
// iPhone Camera + ONNX Runtime Web
// MASK ONLY - NO BOUNDING BOX
// ============================================================

const MODEL_PATH = "./best.onnx";

const INPUT_SIZE = 320;

const CONF_THRESHOLD = 0.60;
const IOU_THRESHOLD = 0.50;
const MASK_THRESHOLD = 0.50;

const INFERENCE_INTERVAL = 100;


// ============================================================
// CLASS NAMES
// ============================================================

const CLASS_NAMES = [
    "CHIEU_TIM",              // ID 0
    "CHUOI_DUOC_DO_RUNG",     // ID 1
    "CHUOI_MO_KET",           // ID 2
    "DAM_BUT_CAM",            // ID 3
    "DAM_BUT_VANG",           // ID 4
    "HONG",                   // ID 5
    "HUYNH_ANH",              // ID 6
    "MAO_GA_DUOI_CONG",       // ID 7
    "MAU_DON_DO",             // ID 8
    "MAU_DON_TRANG",          // ID 9
    "MAU_DON_VANG",           // ID 10
    "MIA_DO",                 // ID 11
    "MO_TRANG",               // ID 12
    "NGOC_BUT"                // ID 13
];


// ============================================================
// CLASS COLORS
// ============================================================

const CLASS_COLORS = [
    [255, 255, 0],    // ID 0  - CHIEU_TIM
    [0, 200, 255],    // ID 1  - CHUOI_DUOC_DO_RUNG
    [255, 165, 0],    // ID 2  - CHUOI_MO_KET
    [0, 255, 0],      // ID 3  - DAM_BUT_CAM
    [200, 80, 255],   // ID 4  - DAM_BUT_VANG
    [255, 0, 0],      // ID 5  - HONG
    [0, 128, 255],    // ID 6  - HUYNH_ANH
    [255, 0, 255],    // ID 7  - MAO_GA_DUOI_CONG
    [128, 255, 0],    // ID 8  - MAU_DON_DO
    [255, 255, 255],  // ID 9  - MAU_DON_TRANG
    [0, 255, 255],    // ID 10 - MAU_DON_VANG
    [128, 0, 255],    // ID 11 - MIA_DO
    [255, 128, 0],    // ID 12 - MO_TRANG
    [0, 255, 128]    // ID 13 - NGOC_BUT
];


// ============================================================
// GLOBAL VARIABLES
// ============================================================

let session = null;
let stream = null;

let running = false;
let inferenceBusy = false;

let lastInferenceTime = 0;

let frameCounter = 0;
let lastFPSUpdate = performance.now();

let outputsLogged = false;


// ============================================================
// DOM
// ============================================================

const video =
    document.getElementById("video");

const overlay =
    document.getElementById("overlay");

const overlayCtx =
    overlay.getContext("2d");

const processingCanvas =
    document.getElementById("processingCanvas");

const processingCtx =
    processingCanvas.getContext(
        "2d",
        {
            willReadFrequently: true
        }
    );

const statusElement =
    document.getElementById("status");

const fpsElement =
    document.getElementById("fps");

const objectsElement =
    document.getElementById("objects");

const detectionsElement =
    document.getElementById("detections");

const loadButton =
    document.getElementById("loadButton");

const cameraButton =
    document.getElementById("cameraButton");

const stopButton =
    document.getElementById("stopButton");


// ============================================================
// STATUS
// ============================================================

function setStatus(text) {

    if (statusElement) {
        statusElement.textContent = text;
    }

    console.log(
        "[SEGMENTATION]",
        text
    );
}


// ============================================================
// LOAD MODEL
// ============================================================

async function loadModel() {

    if (session) {

        setStatus(
            "Segmentation model already loaded"
        );

        return true;
    }

    try {

        setStatus(
            "Loading segmentation model..."
        );

        if (loadButton) {
            loadButton.disabled = true;
        }


        // ----------------------------------------------------
        // TRY WEBGL
        // ----------------------------------------------------

        try {

            session =
                await ort.InferenceSession.create(
                    MODEL_PATH,
                    {
                        executionProviders: [
                            "webgl"
                        ],
                        graphOptimizationLevel:
                            "all"
                    }
                );

            setStatus(
                "Segmentation loaded - WebGL"
            );

        }

        catch (webglError) {

            console.warn(
                "WebGL failed:",
                webglError
            );


            // ------------------------------------------------
            // WASM FALLBACK
            // ------------------------------------------------

            session =
                await ort.InferenceSession.create(
                    MODEL_PATH,
                    {
                        executionProviders: [
                            "wasm"
                        ],
                        graphOptimizationLevel:
                            "all"
                    }
                );

            setStatus(
                "Segmentation loaded - WASM"
            );
        }


        console.log(
            "INPUT NAMES:",
            session.inputNames
        );

        console.log(
            "OUTPUT NAMES:",
            session.outputNames
        );

        console.log(
            "INPUT METADATA:",
            session.inputMetadata
        );

        console.log(
            "OUTPUT METADATA:",
            session.outputMetadata
        );


        return true;

    }

    catch (error) {

        console.error(
            "MODEL ERROR:",
            error
        );

        session = null;

        if (loadButton) {
            loadButton.disabled = false;
        }

        setStatus(
            "Model loading failed"
        );

        alert(
            "Không load được best.onnx.\n\n" +
            "Hãy kiểm tra best.onnx có phải " +
            "YOLO Segmentation model hay không."
        );

        return false;
    }
}


// ============================================================
// START CAMERA
// ============================================================

async function startCamera() {

    if (!session) {

        const loaded =
            await loadModel();

        if (!loaded) {
            return;
        }
    }


    try {

        // ----------------------------------------------------
        // STOP OLD STREAM
        // ----------------------------------------------------

        if (stream) {

            stream
                .getTracks()
                .forEach(
                    track =>
                        track.stop()
                );

            stream = null;
        }


        setStatus(
            "Requesting rear camera..."
        );


        // ----------------------------------------------------
        // IPHONE REAR CAMERA
        // ----------------------------------------------------

        stream =
            await navigator
                .mediaDevices
                .getUserMedia({

                    video: {

                        facingMode: {
                            ideal: "environment"
                        },

                        width: {
                            ideal: 640
                        },

                        height: {
                            ideal: 480
                        },

                        frameRate: {
                            ideal: 30,
                            max: 30
                        }
                    },

                    audio: false
                });


        video.srcObject =
            stream;

        await video.play();

        setupOverlay();

        running = true;
        inferenceBusy = false;
        lastInferenceTime = 0;

        setStatus(
            "Segmentation camera running"
        );

        requestAnimationFrame(
            inferenceLoop
        );

    }

    catch (error) {

        console.error(
            "CAMERA ERROR:",
            error
        );

        setStatus(
            "Camera error"
        );

        alert(
            "Không mở được camera.\n\n" +
            "Hãy dùng HTTPS và cho phép Safari " +
            "truy cập camera."
        );
    }
}


// ============================================================
// STOP CAMERA
// ============================================================

function stopCamera() {

    running = false;
    inferenceBusy = false;

    if (stream) {

        stream
            .getTracks()
            .forEach(
                track =>
                    track.stop()
            );

        stream = null;
    }

    video.srcObject = null;

    overlayCtx.clearRect(
        0,
        0,
        overlay.width,
        overlay.height
    );

    objectsElement.textContent =
        "0";

    fpsElement.textContent =
        "0";

    detectionsElement.textContent =
        "No detection";

    setStatus(
        session
            ? "Model loaded"
            : "Model not loaded"
    );
}


// ============================================================
// OVERLAY
// ============================================================

function setupOverlay() {

    if (
        !video.videoWidth ||
        !video.videoHeight
    ) {
        return;
    }

    overlay.width =
        video.videoWidth;

    overlay.height =
        video.videoHeight;
}


// ============================================================
// PREPROCESS IMAGE
// ============================================================

function preprocess() {

    const videoWidth =
        video.videoWidth;

    const videoHeight =
        video.videoHeight;


    if (
        !videoWidth ||
        !videoHeight
    ) {
        return null;
    }


    processingCanvas.width =
        INPUT_SIZE;

    processingCanvas.height =
        INPUT_SIZE;


    processingCtx.clearRect(
        0,
        0,
        INPUT_SIZE,
        INPUT_SIZE
    );


    // --------------------------------------------------------
    // LETTERBOX
    // --------------------------------------------------------

    const scale =
        Math.min(
            INPUT_SIZE / videoWidth,
            INPUT_SIZE / videoHeight
        );


    const newWidth =
        videoWidth * scale;

    const newHeight =
        videoHeight * scale;


    const offsetX =
        (INPUT_SIZE - newWidth) / 2;

    const offsetY =
        (INPUT_SIZE - newHeight) / 2;


    processingCtx.fillStyle =
        "#808080";

    processingCtx.fillRect(
        0,
        0,
        INPUT_SIZE,
        INPUT_SIZE
    );


    processingCtx.drawImage(
        video,

        0,
        0,
        videoWidth,
        videoHeight,

        offsetX,
        offsetY,
        newWidth,
        newHeight
    );


    const imageData =
        processingCtx.getImageData(
            0,
            0,
            INPUT_SIZE,
            INPUT_SIZE
        );


    const pixels =
        imageData.data;


    const pixelCount =
        INPUT_SIZE *
        INPUT_SIZE;


    const floatData =
        new Float32Array(
            pixelCount * 3
        );


    // --------------------------------------------------------
    // RGB -> CHW
    // --------------------------------------------------------

    for (
        let i = 0;
        i < pixelCount;
        i++
    ) {

        const p =
            i * 4;


        floatData[i] =
            pixels[p] / 255.0;


        floatData[
            pixelCount + i
        ] =
            pixels[p + 1] / 255.0;


        floatData[
            pixelCount * 2 + i
        ] =
            pixels[p + 2] / 255.0;
    }


    return {

        tensor:
            new ort.Tensor(
                "float32",
                floatData,
                [
                    1,
                    3,
                    INPUT_SIZE,
                    INPUT_SIZE
                ]
            ),

        scale,

        offsetX,

        offsetY
    };
}


// ============================================================
// RUN INFERENCE
// ============================================================

async function runInference() {

    const prep =
        preprocess();


    if (!prep) {
        return null;
    }


    const inputName =
        session.inputNames[0];


    const feeds = {};

    feeds[inputName] =
        prep.tensor;


    const results =
        await session.run(
            feeds
        );


    if (!outputsLogged) {

        outputsLogged = true;

        console.log(
            "================================"
        );

        console.log(
            "YOLO SEGMENTATION OUTPUTS"
        );

        console.log(
            "================================"
        );


        for (
            const name
            of Object.keys(results)
        ) {

            console.log(
                name,
                results[name].dims
            );
        }
    }


    const outputs =
        session.outputNames.map(
            name =>
                results[name]
        );


    const detectionOutput =
        findDetectionOutput(
            outputs
        );


    const prototypeOutput =
        findPrototypeOutput(
            outputs
        );


    if (!detectionOutput) {

        throw new Error(
            "Không tìm thấy detection output."
        );
    }


    if (!prototypeOutput) {

        throw new Error(
            "Không tìm thấy MASK prototype output. " +
            "Model có thể không phải segmentation."
        );
    }


    return {

        detectionOutput,

        prototypeOutput,

        scale:
            prep.scale,

        offsetX:
            prep.offsetX,

        offsetY:
            prep.offsetY
    };
}


// ============================================================
// FIND DETECTION OUTPUT
// ============================================================

function findDetectionOutput(outputs) {

    return outputs.find(output => {

        if (!output || !output.dims) {
            return false;
        }

        const d = output.dims;

        if (d.length !== 3 || d[0] !== 1) {
            return false;
        }

        const dim1 = d[1];
        const dim2 = d[2];

        // [1, channels, boxes]
        const channelFirst =
            dim1 > 4 &&
            dim1 <= 256 &&
            dim2 > 100;

        // [1, boxes, channels]
        const boxFirst =
            dim1 > 100 &&
            dim2 > 4 &&
            dim2 <= 256;

        return channelFirst || boxFirst;

    }) || null;
}


// ============================================================
// FIND MASK PROTOTYPE OUTPUT
// ============================================================

function findPrototypeOutput(
    outputs
) {

    return outputs.find(
        output => {

            if (
                !output ||
                !output.dims
            ) {
                return false;
            }


            const d =
                output.dims;


            return (
                d.length === 4 &&
                d[0] === 1 &&
                d[1] >= 4 &&
                d[2] >= 8 &&
                d[3] >= 8
            );
        }
    ) || null;
}


// ============================================================
// POSTPROCESS SEGMENTATION
// ============================================================

function postprocessSegmentation(
    detectionOutput,
    prototypeOutput,
    scale,
    offsetX,
    offsetY
) {
    const detDims = detectionOutput.dims;
    const detData = detectionOutput.data;

    const numBoxes = detDims[1];
    const numChannels = detDims[2];

    const protoDims = prototypeOutput.dims;
    const protoData = prototypeOutput.data;

    const protoChannels = protoDims[1];
    const protoHeight = protoDims[2];
    const protoWidth = protoDims[3];

    console.log("Detection shape:", detDims);
    console.log("Prototype shape:", protoDims);

    // 6 thông tin:
    // x1, y1, x2, y2, confidence, class_id
    // + 32 mask coefficients
    if (numChannels !== 6 + protoChannels) {
        throw new Error(
            `Sai format segmentation: detection=${numChannels}, expected=${6 + protoChannels}`
        );
    }

    const detections = [];

    for (let i = 0; i < numBoxes; i++) {

        const base = i * numChannels;

        // ==========================================
        // TỌA ĐỘ TRÊN ẢNH 320x320
        // ==========================================
        const modelX1 = detData[base + 0];
        const modelY1 = detData[base + 1];
        const modelX2 = detData[base + 2];
        const modelY2 = detData[base + 3];

        const score = detData[base + 4];
        const classId = Math.round(detData[base + 5]);

        if (score < CONF_THRESHOLD) {
            continue;
        }

        if (
            classId < 0 ||
            classId >= CLASS_NAMES.length
        ) {
            continue;
        }

        if (
            modelX2 <= modelX1 ||
            modelY2 <= modelY1
        ) {
            continue;
        }

        // ==========================================
        // CHUYỂN BBOX TỪ 320x320 -> CAMERA
        // ==========================================
        let x1 = (modelX1 - offsetX) / scale;
        let y1 = (modelY1 - offsetY) / scale;
        let x2 = (modelX2 - offsetX) / scale;
        let y2 = (modelY2 - offsetY) / scale;

        x1 = Math.max(
            0,
            Math.min(video.videoWidth, x1)
        );

        y1 = Math.max(
            0,
            Math.min(video.videoHeight, y1)
        );

        x2 = Math.max(
            0,
            Math.min(video.videoWidth, x2)
        );

        y2 = Math.max(
            0,
            Math.min(video.videoHeight, y2)
        );

        // ==========================================
        // LẤY 32 MASK COEFFICIENTS
        // ==========================================
        const maskCoefficients =
            new Float32Array(protoChannels);

        for (let m = 0; m < protoChannels; m++) {
            maskCoefficients[m] =
                detData[base + 6 + m];
        }

        detections.push({
            x1,
            y1,
            x2,
            y2,

            // QUAN TRỌNG:
            // giữ bbox ở hệ tọa độ 320x320
            modelX1,
            modelY1,
            modelX2,
            modelY2,

            score,
            classId,

            maskCoefficients,
            protoData,
            protoWidth,
            protoHeight,
            protoChannels
        });
    }

    // Không cần NMS thêm vì output [1,300,38]
    // đã là output đã xử lý.
    detections.sort(
        (a, b) => b.score - a.score
    );

    // ==========================================
    // TẠO MASK
    // ==========================================
    for (const detection of detections) {
        detection.mask = createMask(detection);
    }

    // ==========================================
    // CHỈ LẤY OBJECT CUỐI CÙNG
    // ==========================================
return detections;
}


// ============================================================
// CREATE MASK
// ============================================================

function createMask(detection) {

    const {
        maskCoefficients,
        protoData,
        protoWidth,
        protoHeight,
        protoChannels,

        modelX1,
        modelY1,
        modelX2,
        modelY2
    } = detection;

    const maskSize =
        protoWidth * protoHeight;

    const mask =
        new Uint8Array(maskSize);

    // ==========================================
    // TỌA ĐỘ BBOX 320x320 -> PROTOTYPE 80x80
    // ==========================================

    const scaleX =
        protoWidth / INPUT_SIZE;

    const scaleY =
        protoHeight / INPUT_SIZE;

    const px1 = Math.max(
        0,
        Math.floor(modelX1 * scaleX)
    );

    const py1 = Math.max(
        0,
        Math.floor(modelY1 * scaleY)
    );

    const px2 = Math.min(
        protoWidth - 1,
        Math.ceil(modelX2 * scaleX)
    );

    const py2 = Math.min(
        protoHeight - 1,
        Math.ceil(modelY2 * scaleY)
    );

    // ==========================================
    // TÍNH MASK
    // ==========================================

    for (let py = 0; py < protoHeight; py++) {

        for (let px = 0; px < protoWidth; px++) {

            // ======================================
            // NGOÀI BBOX -> MASK = 0
            // ======================================

            if (
                px < px1 ||
                px > px2 ||
                py < py1 ||
                py > py2
            ) {
                continue;
            }

            const p =
                py * protoWidth + px;

            let value = 0;

            // ======================================
            // 32 PROTOTYPES × 32 COEFFICIENTS
            // ======================================

            for (
                let c = 0;
                c < protoChannels;
                c++
            ) {

                const protoIndex =
                    c * maskSize + p;

                value +=
                    maskCoefficients[c] *
                    protoData[protoIndex];
            }

            // ======================================
            // SIGMOID
            // ======================================

            const probability =
                1 /
                (1 + Math.exp(-value));

            // ======================================
            // MASK THRESHOLD
            // ======================================

            if (
                probability >= MASK_THRESHOLD
            ) {
                mask[p] = 255;
            }
        }
    }

    return {
        data: mask,
        width: protoWidth,
        height: protoHeight
    };
}

// ============================================================
// IOU
// ============================================================

function calculateIoU(
    a,
    b
) {

    const x1 =
        Math.max(
            a.x1,
            b.x1
        );

    const y1 =
        Math.max(
            a.y1,
            b.y1
        );

    const x2 =
        Math.min(
            a.x2,
            b.x2
        );

    const y2 =
        Math.min(
            a.y2,
            b.y2
        );


    const intersectionWidth =
        Math.max(
            0,
            x2 - x1
        );


    const intersectionHeight =
        Math.max(
            0,
            y2 - y1
        );


    const intersection =
        intersectionWidth *
        intersectionHeight;


    const areaA =
        Math.max(
            0,
            a.x2 - a.x1
        ) *
        Math.max(
            0,
            a.y2 - a.y1
        );


    const areaB =
        Math.max(
            0,
            b.x2 - b.x1
        ) *
        Math.max(
            0,
            b.y2 - b.y1
        );


    const union =
        areaA +
        areaB -
        intersection;


    if (
        union <= 0
    ) {
        return 0;
    }


    return (
        intersection /
        union
    );
}


// ============================================================
// NMS
// ============================================================

function nms(
    boxes,
    threshold
) {

    const sorted =
        [...boxes].sort(
            (
                a,
                b
            ) =>
                b.score -
                a.score
        );


    const selected = [];


    while (
        sorted.length > 0
    ) {

        const current =
            sorted.shift();


        selected.push(
            current
        );


        for (
            let i =
                sorted.length - 1;

            i >= 0;

            i--
        ) {

            const box =
                sorted[i];


            // NMS theo class
            if (
                box.classId !==
                current.classId
            ) {
                continue;
            }


            const iou =
                calculateIoU(
                    current,
                    box
                );


            if (
                iou >=
                threshold
            ) {

                sorted.splice(
                    i,
                    1
                );
            }
        }
    }


    return selected;
}


// ============================================================
// DRAW SEGMENTATION MASK ONLY
// ============================================================

function drawDetections(detections) {

    overlayCtx.clearRect(
        0,
        0,
        overlay.width,
        overlay.height
    );

    const scaleX =
        overlay.width / video.videoWidth;

    const scaleY =
        overlay.height / video.videoHeight;

    for (const detection of detections) {

        drawMask(
            detection,
            scaleX,
            scaleY
        );
    }

    updateDetectionList(detections);

    objectsElement.textContent =
        detections.length;
}


// ============================================================
// DRAW MASK
// ============================================================

function drawMask(
    detection,
    scaleX,
    scaleY
) {
    if (!detection.mask) {
        return;
    }

    const {
        data,
        width,
        height
    } = detection.mask;

    // ==========================================
    // TẠO CANVAS MASK 80x80
    // ==========================================

    const maskCanvas =
        document.createElement("canvas");

    maskCanvas.width = width;
    maskCanvas.height = height;

    const maskCtx =
        maskCanvas.getContext("2d");

    const imageData =
        maskCtx.createImageData(
            width,
            height
        );

    const color =
        CLASS_COLORS[detection.classId] ||
        [255, 255, 0];

    for (
        let i = 0;
        i < data.length;
        i++
    ) {

        if (data[i] === 0) {
            continue;
        }

        const j = i * 4;

        imageData.data[j + 0] = color[0];
        imageData.data[j + 1] = color[1];
        imageData.data[j + 2] = color[2];

        // Độ trong suốt
        imageData.data[j + 3] = 110;
    }

    maskCtx.putImageData(
        imageData,
        0,
        0
    );

    // ==========================================
    // KÍCH THƯỚC LETTERBOX
    // ==========================================

    const modelScale =
        Math.min(
            INPUT_SIZE / video.videoWidth,
            INPUT_SIZE / video.videoHeight
        );

    const resizedWidth =
        video.videoWidth * modelScale;

    const resizedHeight =
        video.videoHeight * modelScale;

    const modelOffsetX =
        (INPUT_SIZE - resizedWidth) / 2;

    const modelOffsetY =
        (INPUT_SIZE - resizedHeight) / 2;

    // ==========================================
    // VÙNG PROTOTYPE CẦN LẤY
    // ==========================================

    const sourceX =
        modelOffsetX /
        INPUT_SIZE *
        width;

    const sourceY =
        modelOffsetY /
        INPUT_SIZE *
        height;

    const sourceW =
        resizedWidth /
        INPUT_SIZE *
        width;

    const sourceH =
        resizedHeight /
        INPUT_SIZE *
        height;

    // ==========================================
    // VẼ MASK
    // ==========================================

    overlayCtx.drawImage(
        maskCanvas,

        sourceX,
        sourceY,
        sourceW,
        sourceH,

        0,
        0,
        overlay.width,
        overlay.height
    );

    // ==========================================
    // LABEL
    // ==========================================

    const label =
        `${CLASS_NAMES[detection.classId]} ` +
        `${(detection.score * 100).toFixed(1)}%`;

    overlayCtx.font =
        "bold 24px Arial";

    overlayCtx.fillStyle =
        "#00ff00";

    const labelX =
        detection.x1 * scaleX;

    const labelY =
        Math.max(
            30,
            detection.y1 * scaleY
        );

    overlayCtx.fillText(
        label,
        labelX,
        labelY
    );
}


// ============================================================
// DETECTION LIST
// ============================================================

function updateDetectionList(
    detections
) {

    if (
        detections.length === 0
    ) {

        detectionsElement.textContent =
            "No detection";

        return;
    }


    detectionsElement.innerHTML =
        detections
            .map(
                detection => {

                    const className =
                        CLASS_NAMES[
                            detection.classId
                        ] ||
                        `Class ${
                            detection.classId
                        }`;


                    const confidence =
                        (
                            detection.score *
                            100
                        ).toFixed(1);


                    const color =
                        CLASS_COLORS[
                            detection.classId
                        ] ||
                        [0, 255, 0];


                    return `

                        <div style="
                            padding:6px 8px;
                            margin-bottom:5px;
                            border-radius:6px;

                            background:rgba(
                                ${color[0]},
                                ${color[1]},
                                ${color[2]},
                                0.15
                            );
                        ">

                            <strong>
                                ${className}
                            </strong>

                            - ${confidence}%

                            - SEGMENTATION

                        </div>

                    `;
                }
            )
            .join("");
}


// ============================================================
// FPS
// ============================================================

function updateFPS() {

    frameCounter++;


    const now =
        performance.now();


    const elapsed =
        now -
        lastFPSUpdate;


    if (
        elapsed >= 1000
    ) {

        const fps =
            frameCounter *
            1000 /
            elapsed;


        fpsElement.textContent =
            fps.toFixed(1);


        frameCounter =
            0;


        lastFPSUpdate =
            now;
    }
}


// ============================================================
// REALTIME LOOP
// ============================================================

async function inferenceLoop(
    timestamp
) {

    if (!running) {
        return;
    }


    if (
        video.readyState >= 2 &&

        timestamp -
        lastInferenceTime >=
        INFERENCE_INTERVAL
    ) {

        lastInferenceTime =
            timestamp;


        if (!inferenceBusy) {

            inferenceBusy =
                true;


            try {

                const result =
                    await runInference();


                if (result) {

                    const detections =
                        postprocessSegmentation(

                            result.detectionOutput,

                            result.prototypeOutput,

                            result.scale,

                            result.offsetX,

                            result.offsetY
                        );


                    // ------------------------------------------------
                    // DRAW MASK ONLY
                    // ------------------------------------------------

                    drawDetections(
                        detections
                    );


                    updateFPS();


                    if (
                        detections.length >
                        0
                    ) {

                        setStatus(
                            `Segmented ${
                                detections.length
                            } object(s)`
                        );

                    }

                    else {

                        setStatus(
                            "Scanning..."
                        );
                    }
                }

            }

            catch (error) {

                console.error(
                    "SEGMENTATION ERROR:",
                    error
                );


                setStatus(
                    "Segmentation error - see Console"
                );

            }

            finally {

                inferenceBusy =
                    false;
            }
        }
    }


    requestAnimationFrame(
        inferenceLoop
    );
}


// ============================================================
// BUTTON EVENTS
// ============================================================

if (loadButton) {

    loadButton.addEventListener(
        "click",
        loadModel
    );
}


if (cameraButton) {

    cameraButton.addEventListener(
        "click",
        startCamera
    );
}


if (stopButton) {

    stopButton.addEventListener(
        "click",
        stopCamera
    );
}


// ============================================================
// VIDEO METADATA
// ============================================================

video.addEventListener(
    "loadedmetadata",
    () => {

        setupOverlay();

    }
);


// ============================================================
// WINDOW RESIZE
// ============================================================

window.addEventListener(
    "resize",
    () => {

        setupOverlay();

    }
);


// ============================================================
// INITIAL STATUS
// ============================================================

setStatus(
    "Ready - load segmentation model"
);