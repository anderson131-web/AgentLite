import { NavLink, Route, Routes } from "react-router-dom";
import NewRun from "./pages/NewRun";
import RunDetail from "./pages/RunDetail";
import Runs from "./pages/Runs";
import Evals from "./pages/Evals";

const navClass = ({ isActive }: { isActive: boolean }) =>
  `px-3 py-1.5 rounded-md text-sm font-medium ${
    isActive ? "bg-slate-900 text-white" : "text-slate-600 hover:bg-slate-200"
  }`;

export default function App() {
  return (
    <div className="min-h-screen">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-5xl items-center gap-6 px-4 py-3">
          <span className="font-semibold tracking-tight">
            Agent<span className="text-indigo-600">Lite</span>
          </span>
          <nav className="flex gap-1">
            <NavLink to="/" end className={navClass}>New run</NavLink>
            <NavLink to="/runs" className={navClass}>Runs</NavLink>
            <NavLink to="/evals" className={navClass}>Evals</NavLink>
          </nav>
        </div>
      </header>
      <main className="mx-auto max-w-5xl px-4 py-6">
        <Routes>
          <Route path="/" element={<NewRun />} />
          <Route path="/runs" element={<Runs />} />
          <Route path="/runs/:id" element={<RunDetail />} />
          <Route path="/evals" element={<Evals />} />
          <Route path="*" element={<p className="text-slate-500">Page not found.</p>} />
        </Routes>
      </main>
    </div>
  );
}
