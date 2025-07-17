let video = document.getElementById("webcam");
let canvas = document.getElementById("snapshot");
let resultDiv = document.getElementById("result");

function startWebcam() {
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            video.srcObject = stream;
        })
        .catch(err => {
            alert("Webcam access denied or not available.");
            console.error(err);
        });
}

function captureImage() {
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(blob => {
        let formData = new FormData();
        formData.append("image", blob, "snapshot.jpg");

        fetch("/predict", {
            method: "POST",
            body: formData,
        })
            .then(res => res.json())
            .then(data => {
                resultDiv.innerText = "Result: " + data.prediction;
            })
            .catch(err => {
                console.error("Prediction error:", err);
                resultDiv.innerText = "Prediction failed.";
            });
    }, "image/jpeg");
}

function uploadImage() {
    let input = document.getElementById("imageUpload");
    let file = input.files[0];
    if (!file) {
        alert("Please select an image to upload.");
        return;
    }

    let formData = new FormData();
    formData.append("image", file);

    fetch("/predict", {
        method: "POST",
        body: formData,
    })
        .then(res => res.json())
        .then(data => {
            resultDiv.innerText = "Result: " + data.prediction;
        })
        .catch(err => {
            console.error("Prediction error:", err);
            resultDiv.innerText = "Prediction failed.";
        });
}
