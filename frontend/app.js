const imageInput = document.getElementById("imageInput");
const imagePreview = document.getElementById("imagePreview");
const predictBtn = document.getElementById("predictBtn");
const submitFeedbackBtn = document.getElementById("submitFeedbackBtn");
const tagsContainer = document.getElementById("tagsContainer");
const newTagInput = document.getElementById("newTagInput");
const newTagsContainer = document.getElementById("newTagsContainer");
const statusMessage = document.getElementById("statusMessage");
const fileNameSpan = document.getElementById("fileName");

let selectedFile = null;
let currentSessionId = null;
let currentTags = [];
let tagStates = {};
let newTags = [];
let feedbackSubmitted = false;

// Initial disabled state
setNewTagsEnabled(false);
submitFeedbackBtn.disabled = true;
predictBtn.disabled = false;

// Show image preview
imageInput.addEventListener("change", function () {
    const file = imageInput.files[0];

    fileNameSpan.textContent = file ? file.name : "";

    if (!file) {
        selectedFile = null;
        imagePreview.style.display = "none";
        imagePreview.src = "";
        return;
    }

    selectedFile = file;

    const reader = new FileReader();
    reader.onload = function (e) {
        imagePreview.src = e.target.result;
        imagePreview.style.display = "block";
    };
    reader.readAsDataURL(file);

    statusMessage.textContent = "";
    tagsContainer.innerHTML = "";
    currentSessionId = null;
    currentTags = [];
    tagStates = {};
    newTags = [];
    feedbackSubmitted = false;

    renderNewTags();
    setNewTagsEnabled(false);
    submitFeedbackBtn.disabled = true;
    predictBtn.disabled = false;
});

// Add new tag on Enter
newTagInput.addEventListener("keydown", function (event) {
    if (event.key !== "Enter") return;
    if (newTagInput.disabled) return;

    event.preventDefault();

    const rawValue = newTagInput.value.trim();
    if (!rawValue) return;

    const normalizedValue = rawValue.toLowerCase();

    // check against already-added new tags
    const existsInNewTags = newTags.some(
        tag => tag.toLowerCase() === normalizedValue
    );

    // check against generated tags already shown in UI
    const existsInCurrentTags = currentTags.some(
        tag => String(tag).toLowerCase() === normalizedValue
    );

    if (existsInNewTags || existsInCurrentTags) {
        alert("Tag already exists");
        newTagInput.value = "";
        return;
    }

    newTags.push(rawValue);
    newTagInput.value = "";
    renderNewTags();
});

// Call /predict
predictBtn.addEventListener("click", async function () {
    if (!selectedFile) {
        alert("Please upload an image first.");
        return;
    }

    statusMessage.textContent = "";
    tagsContainer.innerHTML = "";

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
        const response = await fetch("http://18.225.75.118:8000/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Prediction failed.");
        }

        currentSessionId = data.session_id;
        currentTags = data.combined_tags || [];
        feedbackSubmitted = false;

        renderTags(currentTags);
        setNewTagsEnabled(true);
        submitFeedbackBtn.disabled = false;
    } catch (error) {
        console.error(error);
        statusMessage.textContent = `Error: ${error.message}`;
    }
});

// Call /submit-feedback
submitFeedbackBtn.addEventListener("click", async function () {
    if (!currentSessionId) {
        alert("Please generate tags first.");
        return;
    }

    if (feedbackSubmitted) {
        return;
    }

    const reviewedTags = currentTags.map(tag => ({
        tag: tag,
        status: convertTagStateForBackend(tagStates[tag] || "correct")
    }));

    const payload = {
        session_id: currentSessionId,
        reviewed_tags: reviewedTags,
        new_tags: newTags
    };

    const formData = new FormData();
    formData.append("payload_json", JSON.stringify(payload));

    statusMessage.textContent = "Submitting feedback...";

    try {
        const response = await fetch("http://18.225.75.118:8000/submit-feedback", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                typeof data.detail === "string"
                    ? data.detail
                    : JSON.stringify(data.detail)
            );
        }

        statusMessage.textContent = "Feedback submitted successfully.";
        console.log("Feedback response:", data);

        feedbackSubmitted = true;
        submitFeedbackBtn.disabled = true;
        setNewTagsEnabled(false);
        predictBtn.disabled = true;
    } catch (error) {
        console.error(error);
        statusMessage.textContent = `Error: ${error.message}`;
    }
});

function renderTags(tags) {
    tagsContainer.innerHTML = "";
    tagStates = {};

    if (!tags || tags.length === 0) {
        tagsContainer.innerHTML = "<p>No tags returned.</p>";
        return;
    }

    tags.forEach(tagText => {
        const tagEl = document.createElement("div");
        tagEl.className = "tag correct";
        tagEl.textContent = tagText;

        tagStates[tagText] = "correct";

        tagEl.addEventListener("click", () => {
            if (feedbackSubmitted) return;

            const currentState = tagStates[tagText];

            let nextState;
            if (currentState === "correct") nextState = "partial";
            else if (currentState === "partial") nextState = "incorrect";
            else nextState = "correct";

            tagStates[tagText] = nextState;

            tagEl.classList.remove("correct", "partial", "incorrect");
            tagEl.classList.add(nextState);
        });

        tagsContainer.appendChild(tagEl);
    });
}

function renderNewTags() {
    newTagsContainer.innerHTML = "";

    if (newTags.length > 0) {
        newTagsContainer.classList.add("has-tags");
    } else {
        newTagsContainer.classList.remove("has-tags");
    }

    newTags.forEach(tag => {
        const chip = document.createElement("div");
        chip.className = "new-tag-chip";

        const label = document.createElement("span");
        label.textContent = tag;

        const removeBtn = document.createElement("button");
        removeBtn.className = "remove-tag-btn";
        removeBtn.textContent = "×";

        removeBtn.addEventListener("click", function () {
            if (feedbackSubmitted) return;

            newTags = newTags.filter(item => item !== tag);
            renderNewTags();
        });

        chip.appendChild(label);
        chip.appendChild(removeBtn);
        newTagsContainer.appendChild(chip);
    });
}

function setNewTagsEnabled(enabled) {
    newTagInput.disabled = !enabled;

    if (!enabled) {
        newTagInput.value = "";
    }

    newTagInput.placeholder = enabled
        ? "Type a tag and press Enter"
        : "Generate tags first to add new tags";
}

function convertTagStateForBackend(state) {
    if (state === "correct") return "correct";
    if (state === "partial") return "partially correct";
    return "incorrect";
}