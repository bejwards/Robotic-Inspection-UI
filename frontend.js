document.addEventListener("DOMContentLoaded", function() {

    function display_test_piece() {
        var testImage = document.getElementById('test_piece');
        if (testImage) { 
            if (testImage.style.display === 'none') {
                testImage.style.display = 'block';
                toggleButton.style.backgroundColor = '#003366';
            } else {
                testImage.style.display = 'none';
                toggleButton.style.backgroundColor = '#335a85';
            }
        } else {
            console.error('Element with id "test_piece" not found.');
        }
    }

    document.querySelector("button[onclick='display_test_piece()']").onclick = display_test_piece;
});

function save_image(){
    fetch('/save_image', {
        method: 'POST',
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.text();
    })
    .then(data => {
        
    })
    .catch(error => {
        console.error('There was a problem with the fetch operation:', error);
    });

}

document.addEventListener("DOMContentLoaded", function() {
    document.querySelector("button[onclick='saveImage()']").onclick = saveImage;
});

function producePlot(){
    fetch('/produce_plot')
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.text();
    })
    .then(data => {
        
    })
    .catch(error => {
        console.error('There was a problem with the fetch operation:', error);
    });
}

function Plot(){
    fetch('/plot_function')
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.text();
    })
    .then(data => {
        
    })
    .catch(error => {
        console.error('There was a problem with the fetch operation:', error);
    });
}

const kSlider = document.getElementById("k_slider")

document.addEventListener("DOMContentLoaded", function() {
    if (kSlider){
        console.log("JavaScript code executing");
        kSlider.addEventListener("input", function() {
            const k = kSlider.value;
            const kData = new FormData();
            kData.append("k_slider", k);
    
            fetch("/update_k", {
                method: "POST",
                body: kData
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error("Network response was not ok");
                }
                return response.text();
            })
            .then(data => {
                console.log("Success:", data);  
            })
            .catch(error => {
                console.error("Error updating alpha:", error);
            });
        });
    }
});

const pSlider = document.getElementById("p_slider")

document.addEventListener("DOMContentLoaded", function() {
    if (pSlider){
        console.log("JavaScript code executing");
        pSlider.addEventListener("input", function() {
            const p = pSlider.value;
            const pData = new FormData();
            pData.append("p_slider", p);
    
            fetch("/update_p", {
                method: "POST",
                body: pData
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error("Network response was not ok");
                }
                return response.text();
            })
            .then(data => {
                console.log("Success:", data);  
            })
            .catch(error => {
                console.error("Error updating p:", error);
            });
        });
    }
});


function End_loop(){
    fetch('/end_loop')
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.text();
    })
    .then(data => {
        
    })
    .catch(error => {
        console.error('There was a problem with the fetch operation:', error);
    });
}



window.onload = function() {
    var liveImage = document.getElementById("liveplot");

    function refreshImage() {
        liveImage.src = "/display_live?time=" + new Date().getTime();
    }

    setInterval(refreshImage, 1000);
}

window.onload = function() {
    var liveImage = document.getElementById("plot");

    function refreshImage() {
        liveImage.src = "/display_plot?time=" + new Date().getTime();
    }

    setInterval(refreshImage, 100);
}

document.addEventListener("DOMContentLoaded", function() {

    document.getElementById("k_slider").style.display = "none";
    document.querySelector("label[for='k_slider']").style.display = "none";
    document.getElementById("p_slider").style.display = "none";
    document.querySelector("label[for='p_slider']").style.display = "none";


    function handleThresholdingSelection() {
        var dropdown = document.getElementById("thresholding");
        var selectedValue = dropdown.value;
        
        fetch('/handle_thresholding', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ thresholding: selectedValue }),
        })
        .then(response => response.json())
        .then(data => {
            console.log('Slider k value:', data.k);
            console.log('Slider p value:', data.p);

            document.getElementById("k_slider").value = data.k;
            document.getElementById("p_slider").value = data.p;

            if (selectedValue === "customizable") {
                document.getElementById("k_slider").style.display = "block";
                document.querySelector("label[for='k_slider']").style.display = "block";
                document.getElementById("p_slider").style.display = "block";
                document.querySelector("label[for='p_slider']").style.display = "block";
            } else {
                
                document.getElementById("k_slider").style.display = "none";
                document.querySelector("label[for='k_slider']").style.display = "none";
                document.getElementById("p_slider").style.display = "none";
                document.querySelector("label[for='p_slider']").style.display = "none";
            }

        })
        .catch((error) => {
            console.error('Error:', error);
        });
    }

    document.getElementById("thresholding").addEventListener("change", handleThresholdingSelection);


});

function selectColourmap() {
    var colourdropdown = document.getElementById("colourmap");
    var selectedValue = colourdropdown.value;

    fetch('/select_colourmap', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ colourmap: selectedValue }),
    })
    .then(response => response.text())
    .then(data => {
        console.log('Success:', data);

    })
    .catch((error) => {
        console.error('Error:', error);
    });
}

document.getElementById("colourmap").addEventListener("change", selectColourmap);

function realImaginary() {
    var realSelector = document.getElementById("real_imaginary");
    var selectedValue = realSelector.value;

    fetch('/real_or_img', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ real_imaginary: selectedValue }),
    })
    .then(response => response.text())
    .then(data => {
        console.log('Success:', data);

    })
    .catch((error) => {
        console.error('Error:', error);
    });
}

document.getElementById("real_imaginary").addEventListener("change", realImaginary);