import { Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import Home from './pages/Home'
import Search from './pages/Search'
import About from './pages/About'
import Data from './pages/Data'

export default function App() {
  return (
    <div className="bg-white text-[#1D1D1F]">
      <Navbar />
      <Routes>
        <Route path="/"       element={<Home />}   />
        <Route path="/search" element={<Search />} />
        <Route path="/about"  element={<About />}  />
        <Route path="/data"   element={<Data />}   />
      </Routes>
    </div>
  )
}
