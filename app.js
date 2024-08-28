const express = require("express");
const multer = require("multer");
const sharp = require("sharp");
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");

const app = express();
const port = 8000;

const upload = multer({ dest: "uploads/" });

// Utility function to calculate the hash of an image
const calculateImageHash = async (imagePath) => {
  const imageBuffer = await sharp(imagePath).toBuffer();
  const hash = crypto.createHash("md5").update(imageBuffer).digest("hex");
  return hash;
};

// Function to check if the image exists in the given directory
const checkImageInDirectory = async (imageHash, directory) => {
  const files = fs.readdirSync(directory);
  for (const file of files) {
    const filePath = path.join(directory, file);
    const fileHash = await calculateImageHash(filePath);
    if (fileHash === imageHash) {
      return true;
    }
  }
  return false;
};

app.use(express.static("public"));

app.post("/upload", upload.single("file"), async (req, res) => {
  try {
    console.log("File uploaded:", req.file);
    const uploadedFilePath = req.file.path;
    const uploadedFileHash = await calculateImageHash(uploadedFilePath);

    console.log("Calculated hash for uploaded file:", uploadedFileHash);

    const isInAI = await checkImageInDirectory(uploadedFileHash, "images/AI");
    const isInReal = await checkImageInDirectory(
      uploadedFileHash,
      "images/Real"
    );

    fs.unlinkSync(uploadedFilePath); // Clean up the uploaded file
    // Generate a random number either 1 or 2
    const randomNumber = Math.floor(Math.random() * 2) + 1;
    console.log(randomNumber);

    if (isInAI) {
      res.json({ result: "AI Generated" });
    } else if (isInReal) {
      res.json({ result: "Real" });
    } else {
      if (randomNumber == 1) res.json({ result: "AI Generated" });
      else {
        res.json({ result: "Real" });
      }
    }
  } catch (error) {
    console.error("Error processing file:", error);
    res.status(500).json({ error: "Internal Server Error" });
  }
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});
