import { ApplicationConfig } from '@angular/core';
import { provideRouter }     from '@angular/router';
import { provideIonicAngular, IonicRouteStrategy } from '@ionic/angular/standalone';
import { RouteReuseStrategy } from '@angular/router';

import { routes } from './app.routes';

export const appConfig: ApplicationConfig = {
  providers: [
    { provide: RouteReuseStrategy, useClass: IonicRouteStrategy },
    provideIonicAngular({ mode: 'md' }),
    provideRouter(routes),
  ],
};
