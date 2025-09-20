// server-test.js
const express = require("express");
const fetch = require("node-fetch");
require("dotenv").config();

const app = express();
const PORT = process.env.PORT || 3000;

app.get("/whoami", async (req, res) => {
  try {
    if (!process.env.HF_KEY) {
      return res.status(400).json({ error: "HF_KEY não definido no .env" });
    }

    const response = await fetch("https://huggingface.co/api/whoami", {
      headers: {
        Authorization: `Bearer ${process.env.HF_KEY}`,
      },
    });

    const data = await response.json();
    res.json({
      envHF_KEY: process.env.HF_KEY.startsWith("hf_") ? "OK (hf_...)" : "Inválido",
      hfResponse: data,
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.listen(PORT, () => {
  console.log(`✅ Test server rodando em http://localhost:${PORT}`);
});
