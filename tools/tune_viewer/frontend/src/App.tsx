import { Link, Route, Routes } from "react-router-dom";

import AnalysisPage from "./pages/AnalysisPage";
import ExperimentPage from "./pages/ExperimentPage";
import ExperimentsPage from "./pages/ExperimentsPage";
import TrialPage from "./pages/TrialPage";

export default function App() {
  return (
    <>
      <div className="topbar">
        <div className="brand">
          <Link to="/">CSIRO Tune Viewer</Link>
        </div>
        <div className="muted">Ray Tune results explorer</div>
        <div style={{ flex: 1 }} />
        <Link to="/analysis">Analysis</Link>
      </div>
      <main>
        <Routes>
          <Route path="/" element={<ExperimentsPage />} />
          <Route path="/analysis" element={<AnalysisPage />} />
          <Route path="/experiments/:expName" element={<ExperimentPage />} />
          <Route
            path="/experiments/:expName/trials/:trialDir"
            element={<TrialPage />}
          />
        </Routes>
      </main>
    </>
  );
}

