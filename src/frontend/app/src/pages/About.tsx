const P = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.'
const P2 = 'Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.'

export default function About() {
  return (
    <div className="bg-white pt-24">
      <section className="bg-[#F5F5F7] py-20">
        <div className="max-w-4xl mx-auto px-6">
          <h1 className="text-[#1D1D1F] text-4xl font-bold mb-8">About</h1>
          <p className="text-[#6E6E73] text-lg leading-relaxed mb-6">{P}</p>
          <p className="text-[#6E6E73] text-lg leading-relaxed mb-6">{P2}</p>
          <p className="text-[#6E6E73] text-lg leading-relaxed">{P}</p>
        </div>
      </section>
      <section className="bg-white py-16">
        <div className="max-w-4xl mx-auto px-6">
          <h2 className="text-[#1D1D1F] text-3xl font-bold mb-8">Lorem ipsum</h2>
          <p className="text-[#6E6E73] text-base leading-relaxed mb-6">{P2}</p>
          <p className="text-[#6E6E73] text-base leading-relaxed mb-6">{P}</p>
          <p className="text-[#6E6E73] text-base leading-relaxed">{P2}</p>
        </div>
      </section>
      <section className="bg-[#F5F5F7] py-16">
        <div className="max-w-4xl mx-auto px-6">
          <h2 className="text-[#1D1D1F] text-3xl font-bold mb-8">Dolor sit amet</h2>
          <p className="text-[#6E6E73] text-base leading-relaxed mb-6">{P}</p>
          <p className="text-[#6E6E73] text-base leading-relaxed">{P2}</p>
        </div>
      </section>
    </div>
  )
}
