// Filled in once infra/website/ is deployed and API Gateway gives us a real
// URL (tomorrow). Kept in its own file specifically so that update is a
// one-line change here, not a hunt through script.js.
const CONFIG = {
  API_BASE_URL: "https://gr9q9x9t06.execute-api.us-east-1.amazonaws.com",
  // Filled in by CI only when the streaming endpoint passes its health check.
  // Empty means "no streaming": the page uses the plain /chat endpoint.
  STREAM_URL: "",
  RESUME_URL: "https://anmolbhargava.dev/Anmol_Bhargava_Resume.pdf", // point this at the actual hosted resume file once available
};