import { createBrowserRouter, RouterProvider } from 'react-router-dom'
import { MainLayout } from './layouts/MainLayout'
import { Communicate } from './pages/Communicate'
import { SignsGallery } from './pages/SignsGallery'
import { Contribute } from './pages/Contribute'
import { About } from './pages/About'
import { BackendGate } from './components/BackendGate'
import { ModelProvider } from './model/ModelContext'

const router = createBrowserRouter([
  {
    path: '/',
    element: <MainLayout />,
    children: [
      {
        index: true,
        element: <Communicate />,
      },
      {
        path: 'signs',
        element: <SignsGallery />,
      },
      {
        path: 'contribute',
        element: <Contribute />,
      },
      {
        path: 'about',
        element: <About />,
      },
    ],
  },
])

function App() {
  return (
    <BackendGate>
      <ModelProvider>
        <RouterProvider router={router} />
      </ModelProvider>
    </BackendGate>
  )
}

export default App
