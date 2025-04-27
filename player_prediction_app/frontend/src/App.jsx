/**
 * This file is part of NBA Player Predictor.
 * Copyright (C) 2025 John LaVergne
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY.
 * See the GNU General Public License for more details.
 * <https://www.gnu.org/licenses/>.
 */


import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import HomePage from './pages/HomePage';
import RunHistoryPage from './pages/RunHistoryPage';
import RunDetailsPage from './pages/RunDetailsPage';

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/runs" element={<RunHistoryPage />} />
        <Route path="/runs/:runId" element={<RunDetailsPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Router>
  );
}

